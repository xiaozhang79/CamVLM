from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

from .task import TASK_PROMPT, load_samples, parse_prediction, record_key
from evaluation.udvideoqa.generation import (
    resolve_generation_settings,
    transformers_generation_kwargs,
    vllm_generation_kwargs,
)
from evaluation.udvideoqa.infer import (
    fetch_processor_video_inputs,
    fetch_vllm_multimodal_data,
    tensor_to_vllm_array,
)
from evaluation.udvideoqa.video_processing import _mp4_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CamVLM on CCTV-Anomaly.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--view-mode", choices=["full", "fixed"], default="full")
    parser.add_argument("--min-video-tokens", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=448)
    parser.add_argument("--total-video-tokens", type=int, default=14336)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm-top-p", type=float, default=1.0)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--vllm-max-model-len", type=int, default=21504)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    parser.add_argument("--vllm-max-video-clips", type=int, default=64)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-position", type=int, default=0)
    return parser.parse_args()


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def split_balanced(samples: Sequence[Dict[str, Any]], chunks: int, index: int) -> List[Dict[str, Any]]:
    if chunks <= 0 or not 0 <= index < chunks:
        raise ValueError("invalid chunk selection")
    costs = [0] * chunks
    selected: List[List[Dict[str, Any]]] = [[] for _ in range(chunks)]
    weighted = []
    for sample in samples:
        total_frames, fps = _mp4_metadata(sample["video"])
        weighted.append((max(1, int(math.ceil(total_frames / fps))), sample))
    for cost, sample in sorted(weighted, key=lambda item: (-item[0], item[1]["sample_idx"])):
        target = min(range(chunks), key=lambda idx: (costs[idx], len(selected[idx]), idx))
        selected[target].append(sample)
        costs[target] += cost
    return sorted(selected[index], key=lambda sample: sample["sample_idx"])


def make_messages(sample: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    video_item: Dict[str, Any] = {
        "type": "video",
        "video": sample["video"],
        "fps": args.video_fps,
        "max_frames": args.max_frames,
    }
    if args.view_mode == "fixed":
        video_item.update(
            {
                "window": [1.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0],
                "_resize_viewport_to_source": True,
            }
        )
    pixel_scale = (args.qwen_image_patch_size * 2) ** 2
    video_item.update(
        {
            "min_pixels": args.min_video_tokens * pixel_scale,
            "total_pixels": args.total_video_tokens * pixel_scale,
        }
    )
    content: List[Dict[str, Any]] = [
        video_item,
        {"type": "text", "text": TASK_PROMPT},
    ]
    return [{"role": "user", "content": content}]


def load_vllm(args: argparse.Namespace):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    from vllm import LLM

    return LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.torch_dtype,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        limit_mm_per_prompt={"video": args.vllm_max_video_clips},
    )


def fetch_qwen_video_inputs(
    messages: List[Dict[str, Any]], image_patch_size: int
):
    from qwen_vl_utils import process_vision_info

    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=image_patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    video_metadata = []
    if videos:
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)
    return images, videos, {**video_kwargs, "video_metadata": video_metadata}


def generate_qwen_vllm(llm: Any, processor: Any, messages: List[Dict[str, Any]], settings: Any, args: argparse.Namespace) -> str:
    from vllm import SamplingParams

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.view_mode == "fixed":
        multimodal = fetch_vllm_multimodal_data(
            messages,
            args.total_video_tokens,
            args.qwen_image_patch_size,
        )
    else:
        _, videos, processor_kwargs = fetch_qwen_video_inputs(messages, args.qwen_image_patch_size)
        video_metadata = processor_kwargs["video_metadata"]
        multimodal = None
        if videos:
            multimodal = {
                "video": [
                    (tensor_to_vllm_array(video), {**metadata, "do_sample_frames": False})
                    for video, metadata in zip(videos, video_metadata)
                ]
            }
    request: Dict[str, Any] = {"prompt": text}
    if multimodal is not None:
        request["multi_modal_data"] = multimodal
    output = llm.generate([request], SamplingParams(n=1, **vllm_generation_kwargs(settings)), use_tqdm=False)
    return output[0].outputs[0].text.strip()


def generate_qwen_transformers(model: Any, processor: Any, messages: List[Dict[str, Any]], settings: Any, args: argparse.Namespace) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text += "<|im_start|>assistant\n"
    if args.view_mode == "fixed":
        images, videos, processor_kwargs = fetch_processor_video_inputs(
            messages,
            args.total_video_tokens,
            args.qwen_image_patch_size,
        )
    else:
        images, videos, processor_kwargs = fetch_qwen_video_inputs(messages, args.qwen_image_patch_size)
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **processor_kwargs,
    )
    inputs = {
        key: value.to(model.device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
    generation_kwargs = transformers_generation_kwargs(settings)
    generation_kwargs["pad_token_id"] = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        new_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )[0].split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def generate(model: Any, llm: Any, processor: Any, messages: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    settings = resolve_generation_settings(args.max_new_tokens, args.temperature, args.vllm_top_p)
    if llm is not None:
        return generate_qwen_vllm(llm, processor, messages, settings, args)
    return generate_qwen_transformers(model, processor, messages, settings, args)


def load_qwen_processor(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer
    return processor


def load_qwen_model(model_path: str, torch_dtype: str):
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    ).eval()


def main() -> None:
    args = parse_args()
    if args.video_fps != 2.0:
        raise ValueError("CCTV-Anomaly uses the shared 2 FPS evaluation setting.")
    if args.max_frames < 2:
        raise ValueError("--max-frames must be at least 2.")
    if args.min_video_tokens <= 0:
        raise ValueError("--min-video-tokens must be positive.")
    samples = load_samples(args.data)
    if args.max_samples >= 0:
        samples = samples[: args.max_samples]
    pending_samples = split_balanced(samples, args.num_chunks, args.chunk_idx)
    existing = load_records(args.output_file) if args.resume else []
    if not args.resume:
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text("", encoding="utf-8")
    completed = {record_key(record) for record in existing if not record.get("error")}
    pending_samples = [sample for sample in pending_samples if record_key(sample) not in completed]
    if not pending_samples:
        return

    processor = load_qwen_processor(args.model_path)
    vision = getattr(processor, "video_processor", None) or getattr(processor, "image_processor", None)
    args.qwen_image_patch_size = int(getattr(vision, "patch_size"))
    llm = load_vllm(args) if args.use_vllm else None
    model = None if llm is not None else load_qwen_model(args.model_path, args.torch_dtype)
    for sample in tqdm(pending_samples, desc=f"chunk {args.chunk_idx}/{args.num_chunks}", disable=args.quiet, position=args.progress_position, dynamic_ncols=True):
        try:
            raw = generate(model, llm, processor, make_messages(sample, args), args)
            caption, category = parse_prediction(raw)
            record = {**sample, "raw_model_output": raw, "pred_caption": caption, "pred_category": category, "model_family": "qwen"}
        except Exception as exc:
            record = {**sample, "raw_model_output": "", "pred_caption": "", "pred_category": "", "model_family": "qwen", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[ERROR] sample_idx={sample['sample_idx']}: {record['error']}", flush=True)
        append_record(args.output_file, record)
        if model is not None:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
