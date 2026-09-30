from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from .generation import (
    GenerationSettings,
    resolve_generation_settings,
    transformers_generation_kwargs,
    vllm_generation_kwargs,
)

try:
    from .video_processing import (
        QWEN3_CHAT_TEMPLATE,
        TARGET_FPS,
        TOTAL_TOKENS,
        _build_qwen_video_item,
        _fetch_video_qwen_backend,
        _mp4_metadata,
        _process_qwen_video_items,
        resize_viewport_frames_to_source,
    )
    from training.rl.window_utils import apply_actions, load_model, parse_initial_window, split_final_answer, split_think_action
except ImportError:
    from evaluation.udvideoqa.video_processing import (
        QWEN3_CHAT_TEMPLATE,
        TARGET_FPS,
        TOTAL_TOKENS,
        _build_qwen_video_item,
        _fetch_video_qwen_backend,
        _mp4_metadata,
        _process_qwen_video_items,
        resize_viewport_frames_to_source,
    )
    from training.rl.window_utils import apply_actions, load_model, parse_initial_window, split_final_answer, split_think_action


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = os.environ.get("MODEL_PATH")
DEFAULT_DATA_ROOT = os.environ.get("UDVIDEOQA_DATA_ROOT", str(PROJECT_ROOT / "datasets" / "udvideoqa"))
DEFAULT_QUESTION_FILES = [
    str(Path(DEFAULT_DATA_ROOT) / "Set_20" / "2.37pm_10.1pm_clips_60_annotations.jsonl"),
    str(Path(DEFAULT_DATA_ROOT) / "Set_03" / "2.26pm_10.1mins_clips_annotations.jsonl"),
]
DEFAULT_OUTPUT_FILE = str(PROJECT_ROOT / "eval_results" / "udvideoqa" / "CamVLM-RL" / "predictions.jsonl")
DEFAULT_INITIAL_WINDOW = "0.3333,0.3333,0.6667,0.6667"

ACTION_SYSTEM_TEMPLATE = (
    "You are a traffic video analyst.\n\n"
    "In a continuous video, each second you can only see a local viewport from the current video frame, "
    "not the full frame. Based on the visible content inside the viewport, decide each second whether to "
    "move or zoom the viewport to track the objects, events, and information referred to by the question. "
    "After the video ends, answer the question using the observed viewport views.\n\n"
    "Question:\n{question}\n\n"
    "During each second, output only the action JSON. "
    "Use this format:\n\n"
    "{{\"action\": ...}}\n\n"
    "Available actions: {{\"action\": None}} (no movement); {{\"action\": left/right/up/down, \"offset\": 0-1}} "
    "(move the viewport left/right/up/down, range 0-1); {{\"action\": zoom_in, \"scale\": s}}, where 0<s<1 "
    "(shrink the viewport by the direct scale factor); {{\"action\": zoom_out, \"scale\": s}}, where s>1 "
    "(enlarge the viewport by the direct scale factor). "
    "To apply multiple actions, separate them with semicolons, for example: "
    "{{\"action\": zoom_out, \"scale\": 1.5; \"action\": down, \"offset\": 0.12}}.\n\n"
    "When the user later indicates that the video has ended, stop viewport tracking and answer directly with "
    "<answer>your answer</answer>."
)

FIXED_SYSTEM_TEMPLATE = (
    "You are a traffic video analyst. "
    "Watch the video carefully before answering."
)

FINAL_ANSWER_PROMPT_TEMPLATE = (
    "The video has ended.\n\n"
    "Question:\n{question}\n\n"
    "Output the answer using this format: <answer>your answer</answer>."
)


def build_final_answer_prompt(sample_or_question: Any) -> str:
    if isinstance(sample_or_question, dict):
        question = sample_or_question.get("question", "")
    else:
        question = sample_or_question
    return FINAL_ANSWER_PROMPT_TEMPLATE.format(question=str(question).strip())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def clip_dir_from_annotation_file(path: Path) -> Path:
    clip_dir_name = path.stem
    if clip_dir_name.endswith("_annotations"):
        clip_dir_name = clip_dir_name[: -len("_annotations")]
    clip_dir = path.parent / clip_dir_name
    if not clip_dir.is_dir():
        raise FileNotFoundError(f"Clip directory not found for {path}: {clip_dir}")
    return clip_dir


def canonical_sample(row: Dict[str, Any], annotation_file: Path, row_idx: int) -> Dict[str, Any]:
    clip_dir = clip_dir_from_annotation_file(annotation_file)
    video_file = str(row.get("video_file_path", "")).strip()
    if not video_file:
        raise ValueError(f"{annotation_file}:{row_idx + 1} has empty video_file_path")
    if "_blurred" in Path(video_file).name:
        raise ValueError(f"UDVideoQA video_file_path must not contain '_blurred': {video_file}")
    video_path = clip_dir / video_file
    if not video_path.is_file():
        raise FileNotFoundError(f"Video referenced by JSONL does not exist: {video_path}")
    return {
        "sample_idx": None,
        "id": str(row.get("index") or f"{annotation_file.stem}_{row_idx}"),
        "set_name": annotation_file.parent.name,
        "annotation_file": str(annotation_file),
        "video_file_path": video_file,
        "video": str(video_path),
        "category": str(row.get("category", "")).strip(),
        "sub_category": str(row.get("sub-category", "")).strip(),
        "question": str(row.get("question", "")).strip(),
        "answer": str(row.get("answer", "")).strip(),
    }


def load_samples(question_files: Sequence[str]) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    for question_file in question_files:
        path = Path(question_file)
        rows = load_jsonl(path)
        for row_idx, row in enumerate(rows):
            sample = canonical_sample(row, path, row_idx)
            sample["sample_idx"] = len(samples)
            samples.append(sample)
    return samples


def build_clip_items(video_path: str) -> List[Dict[str, Any]]:
    total_frames, fps = _mp4_metadata(video_path)
    duration = total_frames / fps
    full_seconds = int(math.floor(duration + 1e-6))
    if full_seconds <= 0:
        raise ValueError(f"Video has no complete 1-second segment: {video_path}")
    return [
        {
            "type": "video",
            "video": video_path,
            "video_start": float(second),
            "video_end": float(second + 1),
        }
        for second in range(full_seconds)
    ]


def item_with_window(item: Dict[str, Any], window: Sequence[float]) -> Dict[str, Any]:
    output = dict(item)
    output["window"] = [round(float(value), 4) for value in window]
    return output


def collect_video_items(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for message in messages:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "video":
                items.append(item)
    return items


def assert_video_context(messages: Sequence[Dict[str, Any]], expected: int) -> None:
    actual = len(collect_video_items(messages))
    if actual != expected:
        raise RuntimeError(f"Expected {expected} accumulated video clips, found {actual}.")


def prepare_qwen_video_items(
    messages: Sequence[Dict[str, Any]],
    total_video_tokens: int,
    image_patch_size: int,
) -> List[Dict[str, Any]]:
    video_items = collect_video_items(messages)
    pixel_scale = (image_patch_size * 2) ** 2
    clip_total_pixels = None
    if video_items and total_video_tokens > 0:
        clip_total_pixels = int(total_video_tokens * pixel_scale / len(video_items))
    qwen_items = []
    for item in video_items:
        frames, metadata = _fetch_video_qwen_backend(item)
        if item.get("_resize_viewport_to_source"):
            frames = resize_viewport_frames_to_source(frames, metadata)
        qwen_items.append(_build_qwen_video_item(item, frames, metadata, pixel_scale, clip_total_pixels))
    return qwen_items


def tensor_to_vllm_array(video: torch.Tensor) -> np.ndarray:
    tensor = video.detach().cpu()
    if tensor.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims, got {tuple(tensor.shape)}")
    if tensor.shape[1] in (1, 3):
        tensor = tensor.permute(0, 2, 3, 1).contiguous()
    return tensor.numpy().astype(np.uint8)


def metadata_to_vllm_dict(metadata: Dict[str, Any]) -> Dict[str, Any]:
    value = dict(metadata)
    value["do_sample_frames"] = False
    return value


def fetch_processor_video_inputs(
    messages: Sequence[Dict[str, Any]],
    total_video_tokens: int,
    image_patch_size: int,
):
    qwen_items = prepare_qwen_video_items(messages, total_video_tokens, image_patch_size)
    image_inputs, video_inputs, processor_kwargs = _process_qwen_video_items(qwen_items, image_patch_size)
    return image_inputs, video_inputs, processor_kwargs


def fetch_vllm_multimodal_data(
    messages: Sequence[Dict[str, Any]],
    total_video_tokens: int,
    image_patch_size: int,
):
    qwen_items = prepare_qwen_video_items(messages, total_video_tokens, image_patch_size)
    if not qwen_items:
        return None
    _, video_inputs, processor_kwargs = _process_qwen_video_items(qwen_items, image_patch_size)
    video_metadata = processor_kwargs.get("video_metadata") or []
    return {
        "video": [
            (tensor_to_vllm_array(video), metadata_to_vllm_dict(metadata))
            for video, metadata in zip(video_inputs or [], video_metadata)
        ]
    }


def message(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def system_message(sample: Dict[str, Any], mode: str) -> Dict[str, Any]:
    if mode == "camvlm_action_only":
        template = ACTION_SYSTEM_TEMPLATE
    elif mode == "camvlm_fixed_window":
        template = FIXED_SYSTEM_TEMPLATE
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return message("system", template.format(question=sample["question"]))


def parse_answer_text(output: str) -> str:
    _, answer = split_final_answer(output)
    text = answer or output
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|endoftext\|>.*$", "", text, flags=re.DOTALL)
    return re.sub(r"\s+", " ", text).strip()


def generate_assistant(
    model,
    processor,
    messages: List[Dict[str, Any]],
    generation_settings: GenerationSettings,
    total_video_tokens: int,
    image_patch_size: int,
) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text += "<|im_start|>assistant\n"

    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    image_inputs, video_inputs, processor_kwargs = fetch_processor_video_inputs(
        messages,
        total_video_tokens,
        image_patch_size,
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        do_resize=False,
        **processor_kwargs,
    )
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    generation_kwargs = transformers_generation_kwargs(generation_settings)
    generation_kwargs["pad_token_id"] = pad_token_id
    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return decoded.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def generate_assistant_vllm(
    llm,
    processor,
    messages: List[Dict[str, Any]],
    generation_settings: GenerationSettings,
    total_video_tokens: int,
    image_patch_size: int,
) -> str:
    from vllm import SamplingParams

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    text += "<|im_start|>assistant\n"
    request: Dict[str, Any] = {"prompt": text}
    multimodal = fetch_vllm_multimodal_data(messages, total_video_tokens, image_patch_size)
    if multimodal is not None:
        request["multi_modal_data"] = multimodal
    sampling = SamplingParams(n=1, **vllm_generation_kwargs(generation_settings))
    outputs = llm.generate([request], sampling_params=sampling, use_tqdm=False)
    decoded = outputs[0].outputs[0].text
    return decoded.split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0].strip()


def generate_backend(
    model,
    llm,
    processor,
    messages: List[Dict[str, Any]],
    max_new_tokens: int,
    args: argparse.Namespace,
) -> str:
    generation_settings = resolve_generation_settings(
        max_new_tokens,
        args.temperature,
        args.vllm_top_p,
    )
    if llm is not None:
        return generate_assistant_vllm(
            llm,
            processor,
            messages,
            generation_settings,
            args.total_video_tokens,
            args.qwen_image_patch_size,
        )
    return generate_assistant(
        model,
        processor,
        messages,
        generation_settings,
        args.total_video_tokens,
        args.qwen_image_patch_size,
    )


def apply_tracking_output(window: Tuple[float, float, float, float], output: str) -> Tuple[str, List[Dict[str, Any]], Tuple[float, float, float, float]]:
    _, action_text, actions = split_think_action(output)
    context_action = action_text.strip() or '{"action": None}'
    return context_action, actions, apply_actions(window, actions, scale_is_factor=True)


def build_initial_state(sample_idx: int, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    clip_items = build_clip_items(sample["video"])
    return {
        "sample_idx": sample_idx,
        "sample": sample,
        "clip_items": clip_items,
        "messages": [system_message(sample, args.mode)],
        "answer_video_items": [],
        "window": parse_initial_window(args.initial_window),
        "per_second_process": [],
    }


def append_fixed_or_full_context(state: Dict[str, Any], args: argparse.Namespace) -> None:
    fixed_window = parse_initial_window(args.initial_window)
    for step_idx, item in enumerate(state["clip_items"]):
        video_item = item_with_window(item, fixed_window)
        video_item["_resize_viewport_to_source"] = True
        state["messages"].append({"role": "user", "content": [video_item]})
        state["answer_video_items"].append(dict(video_item))
        process_step = {
            "step_idx": step_idx,
            "video_start": video_item["video_start"],
            "video_end": video_item["video_end"],
            "window": video_item.get("window"),
            "mode": args.mode,
        }
        state["per_second_process"].append(process_step)
    assert_video_context(state["messages"], len(state["clip_items"]))


def run_action_context(model, llm, processor, state: Dict[str, Any], args: argparse.Namespace) -> None:
    for step_idx, item in enumerate(state["clip_items"]):
        current_window = state["window"]
        video_item = item_with_window(item, current_window)
        video_item["_resize_viewport_to_source"] = True
        state["messages"].append({"role": "user", "content": [video_item]})
        state["answer_video_items"].append(dict(video_item))
        assert_video_context(state["messages"], step_idx + 1)
        output = generate_backend(model, llm, processor, state["messages"], args.max_new_tokens, args)
        action_text, actions, next_window = apply_tracking_output(current_window, output)
        state["window"] = next_window
        state["messages"].append(message("assistant", action_text))
        state["per_second_process"].append(
            {
                "step_idx": step_idx,
                "video_start": video_item["video_start"],
                "video_end": video_item["video_end"],
                "window_before": list(current_window),
                "window_after": list(next_window),
                "raw_model_output": output,
                "action_text": action_text,
                "actions": actions,
                "mode": args.mode,
            }
        )


def build_clean_final_answer_messages(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    sample = state["sample"]
    messages = [
        message("system", FIXED_SYSTEM_TEMPLATE),
    ]
    for video_item in state["answer_video_items"]:
        messages.append({"role": "user", "content": [dict(video_item)]})
    messages.append(message("user", build_final_answer_prompt(sample)))
    return messages


def generate_final_answer(model, llm, processor, state: Dict[str, Any], args: argparse.Namespace) -> Tuple[str, str]:
    if args.mode == "camvlm_fixed_window":
        messages = build_clean_final_answer_messages(state)
    else:
        state["messages"].append(message("user", build_final_answer_prompt(state["sample"])))
        messages = state["messages"]
    assert_video_context(messages, len(state["clip_items"]))
    output = generate_backend(model, llm, processor, messages, args.final_max_new_tokens, args)
    return parse_answer_text(output), output


def build_output_record(state: Dict[str, Any], pred_answer: str, final_output: str) -> Dict[str, Any]:
    sample = state["sample"]
    return {
        "sample_idx": state["sample_idx"],
        "id": sample["id"],
        "set_name": sample["set_name"],
        "annotation_file": sample["annotation_file"],
        "video_file_path": sample["video_file_path"],
        "video": sample["video"],
        "category": sample["category"],
        "sub_category": sample["sub_category"],
        "question": sample["question"],
        "gt_answer": sample["answer"],
        "pred_answer": pred_answer,
        "final_model_output": final_output,
        "per_second_process": state["per_second_process"],
    }


def run_sample(model, llm, processor, sample_idx: int, sample: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    state = build_initial_state(sample_idx, sample, args)
    if args.mode == "camvlm_action_only":
        run_action_context(model, llm, processor, state, args)
    else:
        append_fixed_or_full_context(state, args)
    pred_answer, final_output = generate_final_answer(model, llm, processor, state, args)
    record = build_output_record(state, pred_answer, final_output)
    record["mode"] = args.mode
    record["model_family"] = "qwen"
    return record


def record_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return str(record.get("id", "")), str(record.get("video_file_path", ""))


def split_chunk(indexed_samples: Sequence[Tuple[int, Dict[str, Any]]], num_chunks: int, chunk_idx: int) -> List[Tuple[int, Dict[str, Any]]]:
    if num_chunks <= 0:
        raise ValueError("--num_chunks must be positive")
    if chunk_idx < 0 or chunk_idx >= num_chunks:
        raise ValueError("--chunk_idx must satisfy 0 <= chunk_idx < num_chunks")
    chunk_size = int(math.ceil(len(indexed_samples) / num_chunks)) if indexed_samples else 0
    start = chunk_idx * chunk_size
    return list(indexed_samples[start : start + chunk_size])


def estimate_clip_count(sample: Dict[str, Any]) -> int:
    return len(build_clip_items(sample["video"]))


def split_balanced_chunk(indexed_samples: Sequence[Tuple[int, Dict[str, Any]]], num_chunks: int, chunk_idx: int, quiet: bool) -> List[Tuple[int, Dict[str, Any]]]:
    weighted = [(estimate_clip_count(sample), sample_idx, sample) for sample_idx, sample in indexed_samples]
    chunks: List[List[Tuple[int, Dict[str, Any]]]] = [[] for _ in range(num_chunks)]
    costs = [0] * num_chunks
    for cost, sample_idx, sample in sorted(weighted, key=lambda item: (-item[0], item[1])):
        target = min(range(num_chunks), key=lambda idx: (costs[idx], len(chunks[idx]), idx))
        chunks[target].append((sample_idx, sample))
        costs[target] += cost
    selected = sorted(chunks[chunk_idx], key=lambda item: item[0])
    if not quiet:
        print(
            "Balanced chunk costs by 1-second clips: "
            + ", ".join(f"{idx}:{cost}" for idx, cost in enumerate(costs)),
            flush=True,
        )
        print(
            f"Selected chunk {chunk_idx}/{num_chunks}: "
            f"{len(selected)} samples, {costs[chunk_idx]} clips.",
            flush=True,
        )
    return selected


def select_samples(samples: Sequence[Dict[str, Any]], sample_idx: int, max_samples: int) -> List[Tuple[int, Dict[str, Any]]]:
    if sample_idx >= 0:
        if sample_idx >= len(samples):
            raise IndexError(f"--sample_idx {sample_idx} is out of range for {len(samples)} samples")
        return [(sample_idx, samples[sample_idx])]
    indexed = list(enumerate(samples))
    return indexed[:max_samples] if max_samples >= 0 else indexed


def load_output_records(output_file: str) -> List[Dict[str, Any]]:
    path = Path(output_file)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_output_records(output_file: str, records: Sequence[Dict[str, Any]]) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=False)
            handle.write("\n")


def append_output_record(output_file: str, record: Dict[str, Any]) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
        handle.write("\n")


def make_error_record(
    sample_idx: int,
    sample: Dict[str, Any],
    exc: Exception,
    mode: str,
    model_family: str,
) -> Dict[str, Any]:
    return {
        "sample_idx": sample_idx,
        "id": sample.get("id", ""),
        "set_name": sample.get("set_name", ""),
        "annotation_file": sample.get("annotation_file", ""),
        "video_file_path": sample.get("video_file_path", ""),
        "video": sample.get("video", ""),
        "category": sample.get("category", ""),
        "sub_category": sample.get("sub_category", ""),
        "question": sample.get("question", ""),
        "gt_answer": sample.get("answer", ""),
        "pred_answer": "",
        "final_model_output": "",
        "per_second_process": [],
        "mode": mode,
        "model_family": model_family,
        "error": f"{type(exc).__name__}: {exc}",
    }


def load_vllm_backend(args: argparse.Namespace):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment.") from exc
    return LLM(
        model=args.vllm_model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        dtype=args.torch_dtype,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        max_model_len=args.vllm_max_model_len,
        max_num_seqs=args.vllm_max_num_seqs,
        limit_mm_per_prompt={"video": args.vllm_max_video_clips},
    )


def run_inference(args: argparse.Namespace) -> None:
    final_generation_settings = resolve_generation_settings(
        args.final_max_new_tokens,
        args.temperature,
        args.vllm_top_p,
    )
    if not args.quiet:
        print(f"Final generation settings: {final_generation_settings}", flush=True)
    samples = load_samples(args.question_files)
    indexed_samples = select_samples(samples, args.sample_idx, args.max_samples)
    if args.sample_idx >= 0 or args.disable_balanced_chunks:
        indexed_samples = split_chunk(indexed_samples, args.num_chunks, args.chunk_idx)
    else:
        indexed_samples = split_balanced_chunk(indexed_samples, args.num_chunks, args.chunk_idx, args.quiet)

    output_records = load_output_records(args.output_file) if args.resume else []
    if not args.resume:
        save_output_records(args.output_file, output_records)
    completed = {record_key(record) for record in output_records}
    pending = [(idx, sample) for idx, sample in indexed_samples if record_key(sample) not in completed]
    if not pending:
        if not args.quiet:
            print(f"No pending samples for chunk {args.chunk_idx}/{args.num_chunks}.")
        return

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor.chat_template = QWEN3_CHAT_TEMPLATE
    vision_processor = getattr(processor, "video_processor", None) or getattr(processor, "image_processor", None)
    args.qwen_image_patch_size = int(getattr(vision_processor, "patch_size"))
    llm = load_vllm_backend(args) if args.use_vllm else None
    model = None if llm is not None else load_model(args.model_path, args.torch_dtype)
    if model is not None:
        model.eval()

    for sample_idx, sample in tqdm(
        pending,
        desc=f"chunk {args.chunk_idx}/{args.num_chunks}",
        disable=args.quiet,
        position=args.progress_position,
        dynamic_ncols=True,
    ):
        try:
            record = run_sample(model, llm, processor, sample_idx, sample, args)
        except Exception as exc:
            record = make_error_record(sample_idx, sample, exc, args.mode, "qwen")
            print(f"[ERROR] sample_idx={sample_idx}: {record['error']}", flush=True)
        append_output_record(args.output_file, record)
        if model is not None:
            torch.cuda.empty_cache()


def cleanup_torch_distributed() -> None:
    distributed = getattr(torch, "distributed", None)
    if distributed is None:
        return
    if not distributed.is_available() or not distributed.is_initialized():
        return
    try:
        distributed.destroy_process_group()
    except Exception as exc:
        print(f"[WARNING] Failed to destroy torch distributed process group: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CamVLM on UDVideoQA Set20/Set03.")
    parser.add_argument("--mode", choices=["camvlm_action_only", "camvlm_fixed_window"], required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--vllm_model_path", default=None)
    parser.add_argument("--question_files", nargs="+", default=DEFAULT_QUESTION_FILES)
    parser.add_argument("--output_file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--initial_window", default=DEFAULT_INITIAL_WINDOW)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--final_max_new_tokens", type=int, default=512)
    parser.add_argument("--total_video_tokens", type=int, default=TOTAL_TOKENS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--torch_dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.32)
    parser.add_argument("--vllm_max_model_len", type=int, default=21504)
    parser.add_argument("--vllm_max_num_seqs", type=int, default=1)
    parser.add_argument("--vllm_max_video_clips", type=int, default=64)
    parser.add_argument("--vllm_top_p", type=float, default=1.0)
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--disable_balanced_chunks", action="store_true")
    parser.add_argument("--sample_idx", type=int, default=-1)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--progress_position", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path:
        raise ValueError("Set MODEL_PATH or pass --model_path to select a model checkpoint.")
    try:
        if args.vllm_model_path is None:
            args.vllm_model_path = args.model_path
        parse_initial_window(args.initial_window)
        if args.max_new_tokens <= 0 or args.final_max_new_tokens <= 0:
            raise ValueError("Token limits must be positive")
        if args.total_video_tokens <= 0:
            raise ValueError("--total_video_tokens must be positive")
        run_inference(args)
    finally:
        cleanup_torch_distributed()


if __name__ == "__main__":
    main()
