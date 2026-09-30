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
from transformers import AutoProcessor, AutoTokenizer

from .task import TASK_PROMPT, load_samples, parse_prediction, record_key
from training.rl.window_utils import load_model, parse_initial_window
from evaluation.udvideoqa.generation import (
    GenerationSettings,
    resolve_generation_settings,
    transformers_generation_kwargs,
    vllm_generation_kwargs,
)
from evaluation.udvideoqa.infer import apply_tracking_output, tensor_to_vllm_array
from evaluation.udvideoqa.video_processing import (
    QWEN3_CHAT_TEMPLATE,
    _build_qwen_video_item,
    _fetch_video_qwen_backend,
    _mp4_metadata,
    _process_qwen_video_items,
    resize_viewport_frames_to_source,
)


DEFAULT_CAMVLM_MODEL = os.environ.get("CAMVLM_MODEL_PATH")
DEFAULT_INITIAL_WINDOW = "0.3333,0.3333,0.6667,0.6667"

ACTION_SYSTEM_TEMPLATE = (
    "You are a security video analyst.\n\n"
    "In a continuous video, at each observed time step you can only see a local view from the current video "
    "frame, not the full frame. Based on the visible content, decide whether to move or zoom the view to "
    "keep the people involved in the primary anomalous event, the objects they interact with, and their "
    "actions and interactions clearly visible across time while preserving enough surrounding context to "
    "recognize the event. If none of them is visible, explore adjacent areas or zoom out instead of treating "
    "the current view as complete. Do not zoom in so tightly that relevant people, objects, or interactions "
    "are lost. Do not output a category or "
    "description until the video has ended. After the video ends, complete the final task using the observed "
    "views.\n\n"
    "During each observed time step, output only the action JSON. Use this format:\n\n"
    "{{\"action\": ...}}\n\n"
    "Available actions: {{\"action\": None}} (no movement); {{\"action\": left/right/up/down, \"offset\": 0-1}} "
    "(move the view left/right/up/down, range 0-1); {{\"action\": zoom_in, \"scale\": s}}, where 0<s<1 "
    "(shrink the view by the direct scale factor); {{\"action\": zoom_out, \"scale\": s}}, where s>1 "
    "(enlarge the view by the direct scale factor). To apply multiple actions, separate them with semicolons, "
    "for example: {{\"action\": zoom_out, \"scale\": 1.5}}; {{\"action\": down, \"offset\": 0.12}}.\n\n"
    "When the user later indicates that the video has ended, stop tracking and follow the final "
    "task instructions."
)

FINAL_ANSWER_PROMPT = f"The video has ended. Stop tracking.\n\n{TASK_PROMPT}"
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CamVLM dynamic tracking on CCTV-Anomaly.")
    parser.add_argument("--mode", choices=["camvlm_action_only"], required=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--min-video-tokens", type=int, default=64)
    parser.add_argument("--total-video-tokens", type=int, default=14336)
    parser.add_argument("--max-action-turns", type=int, default=224)
    parser.add_argument("--initial-window", default=DEFAULT_INITIAL_WINDOW)
    parser.add_argument("--max-action-new-tokens", type=int, default=1024)
    parser.add_argument("--final-max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm-top-p", type=float, default=1.0)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--vllm-max-model-len", type=int, default=24576)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    parser.add_argument("--vllm-max-video-clips", type=int, default=224)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress-position", type=int, default=0)
    return parser.parse_args()


def text_message(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_record(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def action_start_seconds(full_seconds: int, max_action_turns: int) -> List[int]:
    if full_seconds <= 0:
        return []
    if full_seconds <= max_action_turns:
        return list(range(full_seconds))
    starts = [int(round(index * (full_seconds - 1) / (max_action_turns - 1))) for index in range(max_action_turns)]
    if len(set(starts)) != max_action_turns or starts != sorted(starts):
        raise RuntimeError("Uniform action-turn sampling did not produce unique chronological seconds.")
    return starts


def build_action_clip_items(
    video_path: str,
    max_action_turns: int,
) -> List[Dict[str, Any]]:
    total_frames, fps = _mp4_metadata(video_path)
    full_seconds = int(math.floor(total_frames / fps + 1e-6))
    starts = action_start_seconds(full_seconds, max_action_turns)
    if not starts:
        raise ValueError(f"Video has no complete 1-second segment: {video_path}")
    return [
        {
            "type": "video",
            "video": video_path,
            "video_start": float(second),
            "video_end": float(second + 1),
            "fps": 2.0,
            "max_frames": 2,
        }
        for second in starts
    ]


def split_balanced(samples: Sequence[Dict[str, Any]], chunks: int, index: int, max_action_turns: int) -> List[Dict[str, Any]]:
    if chunks <= 0 or not 0 <= index < chunks:
        raise ValueError("invalid chunk selection")
    costs = [0] * chunks
    selected: List[List[Dict[str, Any]]] = [[] for _ in range(chunks)]
    weighted = []
    for sample in samples:
        total_frames, fps = _mp4_metadata(sample["video"])
        full_seconds = int(math.floor(total_frames / fps + 1e-6))
        weighted.append((min(max(full_seconds, 1), max_action_turns), sample))
    for cost, sample in sorted(weighted, key=lambda item: (-item[0], item[1]["sample_idx"])):
        target = min(range(chunks), key=lambda idx: (costs[idx], len(selected[idx]), idx))
        selected[target].append(sample)
        costs[target] += cost
    return sorted(selected[index], key=lambda sample: sample["sample_idx"])


def collect_video_items(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        item
        for message in messages
        for item in message.get("content", [])
        if isinstance(item, dict) and item.get("type") == "video"
    ]


def action_video_items(
    messages: Sequence[Dict[str, Any]],
    image_patch_size: int,
    total_video_tokens: int,
    min_video_tokens: int,
) -> List[Dict[str, Any]]:
    video_items = collect_video_items(messages)
    if not video_items:
        return []
    if total_video_tokens < len(video_items) * min_video_tokens:
        raise ValueError("Total video token budget cannot cover the minimum for every video item.")
    pixel_scale = (image_patch_size * 2) ** 2
    per_video_tokens = max(min_video_tokens, total_video_tokens // len(video_items))
    clip_total_pixels = per_video_tokens * pixel_scale
    prepared = []
    for item in video_items:
        if "_predecoded_frames" in item:
            frames = list(item["_predecoded_frames"])
            metadata = dict(item["_predecoded_metadata"])
        else:
            frames, metadata = _fetch_video_qwen_backend(item)
            if item.get("_resize_viewport_to_source"):
                frames = resize_viewport_frames_to_source(frames, metadata)
        qwen_item = _build_qwen_video_item(item, frames, metadata, pixel_scale, clip_total_pixels)
        prepared.append(qwen_item)
    return prepared


def action_processor_inputs(
    messages: Sequence[Dict[str, Any]], image_patch_size: int, total_video_tokens: int, min_video_tokens: int
):
    return _process_qwen_video_items(
        action_video_items(messages, image_patch_size, total_video_tokens, min_video_tokens), image_patch_size
    )


def action_vllm_multimodal_data(
    messages: Sequence[Dict[str, Any]], image_patch_size: int, total_video_tokens: int, min_video_tokens: int
):
    prepared = action_video_items(messages, image_patch_size, total_video_tokens, min_video_tokens)
    if not prepared:
        return None
    _, video_inputs, processor_kwargs = _process_qwen_video_items(prepared, image_patch_size)
    return {
        "video": [
            (tensor_to_vllm_array(video), {**metadata, "do_sample_frames": False})
            for video, metadata in zip(video_inputs or [], processor_kwargs.get("video_metadata") or [])
        ]
    }


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


def generation_settings(args: argparse.Namespace, max_new_tokens: int) -> GenerationSettings:
    return resolve_generation_settings(
        max_new_tokens,
        args.temperature,
        args.vllm_top_p,
    )


def generate_transformers(model: Any, processor: Any, messages: List[Dict[str, Any]], settings: GenerationSettings, args: argparse.Namespace) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text += "<|im_start|>assistant\n"
    images, videos, processor_kwargs = action_processor_inputs(
        messages, args.qwen_image_patch_size, args.total_video_tokens, args.min_video_tokens
    )
    inputs = processor(
        text=[text],
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **processor_kwargs,
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    kwargs = transformers_generation_kwargs(settings)
    kwargs["pad_token_id"] = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0].split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def generate_vllm(llm: Any, processor: Any, messages: List[Dict[str, Any]], settings: GenerationSettings, args: argparse.Namespace) -> str:
    from vllm import SamplingParams

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text += "<|im_start|>assistant\n"
    request: Dict[str, Any] = {"prompt": text}
    multimodal = action_vllm_multimodal_data(
        messages, args.qwen_image_patch_size, args.total_video_tokens, args.min_video_tokens
    )
    if multimodal is not None:
        request["multi_modal_data"] = multimodal
    output = llm.generate([request], SamplingParams(n=1, **vllm_generation_kwargs(settings)), use_tqdm=False)
    return output[0].outputs[0].text.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def generate(model: Any, llm: Any, processor: Any, messages: List[Dict[str, Any]], max_new_tokens: int, args: argparse.Namespace) -> str:
    settings = generation_settings(args, max_new_tokens)
    if llm is not None:
        return generate_vllm(llm, processor, messages, settings, args)
    return generate_transformers(model, processor, messages, settings, args)


def tracking_system_message(action_history: Sequence[str] = ()) -> Dict[str, Any]:
    prompt = ACTION_SYSTEM_TEMPLATE
    if action_history:
        prompt += "\n\nPrevious action JSONs, in chronological order:\n" + "\n".join(action_history)
    return text_message("system", prompt)


def run_sample(model: Any, llm: Any, processor: Any, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    clip_items = build_action_clip_items(
        sample["video"], args.max_action_turns
    )
    state_messages = [tracking_system_message()]
    action_history: List[str] = []
    per_second_process = []
    window = parse_initial_window(args.initial_window)

    for step_idx, item in enumerate(clip_items):
        current_window = window
        video_item = dict(item)
        video_item["window"] = [round(value, 4) for value in current_window]
        video_item["_resize_viewport_to_source"] = True
        state_messages.append({"role": "user", "content": [video_item]})
        tracking_messages = [
            tracking_system_message(action_history),
            {"role": "user", "content": [video_item]},
        ]
        output = generate(model, llm, processor, tracking_messages, args.max_action_new_tokens, args)
        action_text, actions, window = apply_tracking_output(current_window, output)
        state_messages.append(text_message("assistant", action_text))
        action_history.append(action_text)
        per_second_process.append(
            {
                "step_idx": step_idx,
                "source_second": int(item["video_start"]),
                "video_start": item["video_start"],
                "video_end": item["video_end"],
                "window_before": list(current_window),
                "window_after": list(window),
                "raw_model_output": output,
                "action_text": action_text,
                "actions": actions,
            }
        )

    state_messages.append(text_message("user", FINAL_ANSWER_PROMPT))
    final_messages = state_messages
    final_output = generate(model, llm, processor, final_messages, args.final_max_new_tokens, args)
    caption, category = parse_prediction(final_output)
    return {
        **sample,
        "raw_model_output": final_output,
        "pred_caption": caption,
        "pred_category": category,
        "mode": args.mode,
        "model_family": "qwen",
        "clip_seconds": len(clip_items),
        "action_turn_count": len(clip_items),
        "per_second_process": per_second_process,
    }


def load_processor(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer
    processor.chat_template = QWEN3_CHAT_TEMPLATE
    return processor


def cleanup_torch_distributed() -> None:
    distributed = getattr(torch, "distributed", None)
    if distributed is not None and distributed.is_available() and distributed.is_initialized():
        distributed.destroy_process_group()


def main() -> None:
    args = parse_args()
    try:
        if args.model_path is None:
            args.model_path = DEFAULT_CAMVLM_MODEL
        if not args.model_path:
            raise ValueError("Set CAMVLM_MODEL_PATH or pass --model-path.")
        if args.video_fps != 2.0:
            raise ValueError("CCTV action evaluation uses the shared 2 FPS setting.")
        if args.min_video_tokens <= 0 or args.total_video_tokens <= 0:
            raise ValueError("Video token limits must be positive.")
        if args.max_action_turns <= 0:
            raise ValueError("--max-action-turns must be positive.")
        if args.total_video_tokens < args.max_action_turns * args.min_video_tokens:
            raise ValueError("--total-video-tokens must cover --max-action-turns times --min-video-tokens.")
        if args.vllm_max_video_clips < args.max_action_turns:
            raise ValueError("--vllm-max-video-clips must be at least --max-action-turns.")
        parse_initial_window(args.initial_window)

        samples = load_samples(args.data)
        if args.max_samples >= 0:
            samples = samples[: args.max_samples]
        pending_samples = split_balanced(samples, args.num_chunks, args.chunk_idx, args.max_action_turns)
        existing = load_records(args.output_file) if args.resume else []
        if not args.resume:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            args.output_file.write_text("", encoding="utf-8")
        completed = {record_key(record) for record in existing if not record.get("error")}
        pending_samples = [sample for sample in pending_samples if record_key(sample) not in completed]
        if not pending_samples:
            return

        processor = load_processor(args.model_path)
        vision_processor = getattr(processor, "video_processor", None) or getattr(processor, "image_processor", None)
        args.qwen_image_patch_size = int(getattr(vision_processor, "patch_size"))
        llm = load_vllm(args) if args.use_vllm else None
        model = None if llm is not None else load_model(args.model_path, args.torch_dtype).eval()

        for sample in tqdm(pending_samples, desc=f"chunk {args.chunk_idx}/{args.num_chunks}", disable=args.quiet, position=args.progress_position, dynamic_ncols=True):
            try:
                record = run_sample(model, llm, processor, sample, args)
            except Exception as exc:
                record = {
                    **sample,
                    "raw_model_output": "",
                    "pred_caption": "",
                    "pred_category": "",
                    "mode": args.mode,
                    "model_family": "qwen",
                    "action_turn_count": 0,
                    "per_second_process": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[ERROR] sample_idx={sample['sample_idx']}: {record['error']}", flush=True)
            append_record(args.output_file, record)
            if model is not None:
                torch.cuda.empty_cache()
    finally:
        cleanup_torch_distributed()


if __name__ == "__main__":
    main()
