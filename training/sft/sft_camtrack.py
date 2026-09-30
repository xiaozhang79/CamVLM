import copy
import importlib.util
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from PIL import Image
from qwen_vl_utils import vision_process as qwen_vision_process
from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    ScriptArguments,
    TrlParser,
)
from transformers import AutoModelForVision2Seq, AutoProcessor

try:
    from transformers import AutoModelForImageTextToText, Qwen3VLForConditionalGeneration
except ImportError:
    AutoModelForImageTextToText = None
    Qwen3VLForConditionalGeneration = None

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


TARGET_FPS = float(os.environ.get("CAMTRACK_SFT_FPS", "2"))
DEFAULT_MEVIS_FPS = float(os.environ.get("CAMTRACK_MEVIS_FPS", "6"))
DEFAULT_YOUTUBE_VOS_FPS = float(os.environ.get("CAMTRACK_YOUTUBE_VOS_FPS", "6"))
TOTAL_TOKENS = int(os.environ.get("CAMTRACK_TOTAL_TOKENS", "14336"))
MIN_TOKENS = int(os.environ.get("CAMTRACK_MIN_TOKENS", "64"))
MAX_FRAMES = int(os.environ.get("CAMTRACK_MAX_FRAMES", "448"))
QWEN3_IMAGE_PATCH_SIZE = 16
QWEN3_VIDEO_PIXEL_SCALE = (QWEN3_IMAGE_PATCH_SIZE * qwen_vision_process.SPATIAL_MERGE_SIZE) ** 2
JPEG_READ_WORKERS = max(1, int(os.environ.get("CAMTRACK_JPEG_READ_WORKERS", "4")))
processor = None


def _load_camvlm_prompts_module():
    prompts_path = SRC_ROOT / "training" / "rl" / "prompts.py"
    spec = importlib.util.spec_from_file_location("camvlm_rl_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CamVLM prompts from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


camvlm_prompts = _load_camvlm_prompts_module()
QWEN3_CAMTRACK_CHAT_TEMPLATE = camvlm_prompts.QWEN3_CAMTRACK_CHAT_TEMPLATE


def _load_model(model_name_or_path: str, model_kwargs: Dict[str, Any]):
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
    return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **model_kwargs)


@dataclass
class SFTFreezeConfig:
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Freeze the language model and lm_head when set."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Freeze the visual tower when set."},
    )
    freeze_merger: bool = field(
        default=False,
        metadata={"help": "Freeze the visual merger when set."},
    )


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def _get_visual_tower(model):
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual

    model_core = getattr(model, "model", None)
    visual = getattr(model_core, "visual", None)
    if visual is not None:
        return visual

    raise ValueError("Could not locate the visual tower as model.visual or model.model.visual.")


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = _get_visual_tower(model)
    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = vision_tower.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger = getattr(vision_tower, "merger", None)
    if merger is None:
        raise ValueError("Could not locate the visual merger as visual.merger.")
    merger_params = merger.parameters()
    set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("Could not locate the language-model head as model.lm_head.")
    set_requires_grad(lm_head.parameters(), not training_args.freeze_llm)

    llm = getattr(model, "model", None)
    if llm is None:
        raise ValueError("Could not locate the language model/core as model.model.")
    llm_params = llm.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def _resolve_compute_dtype(training_args) -> torch.dtype:
    return (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )


def _print_trainable_parameters(model, freeze_config) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable parameters: {trainable:,}/{total:,} "
        f"({100.0 * trainable / total:.2f}%); "
        f"freeze_llm={freeze_config.freeze_llm}, "
        f"freeze_vision_tower={freeze_config.freeze_vision_tower}, "
        f"freeze_merger={freeze_config.freeze_merger}",
        flush=True,
    )


def _distributed_barrier_with_device() -> None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if torch.cuda.is_available():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else:
        torch.distributed.barrier()


def _token_ids(text: str) -> List[int]:
    return processor.tokenizer(text, add_special_tokens=False)["input_ids"]


def _get_question(example: Dict[str, Any]) -> str:
    question = example.get("Question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("QA all-actions sample is missing a non-empty top-level Question.")
    return question.strip()


def _build_final_answer_prompt_message() -> Dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": camvlm_prompts.build_qa_final_prompt()}],
    }


def _is_video_item(item: Dict[str, Any]) -> bool:
    return item.get("type") == "video" and isinstance(item.get("video"), str) and bool(item["video"].strip())


def _is_complete_second_video_item(item: Dict[str, Any]) -> bool:
    try:
        return float(item.get("video_end", 0.0)) - float(item.get("video_start", 0.0)) >= 1.0 - 1e-9
    except (TypeError, ValueError):
        return False


def _is_final_answer_message(message: Dict[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    for item in message.get("content", []):
        if item.get("type") == "text" and item.get("text", "").strip().startswith("<answer>"):
            return True
    return False


def _is_final_answer_prompt_message(message: Dict[str, Any]) -> bool:
    if message.get("role") != "user":
        return False
    for item in message.get("content", []):
        if (
            item.get("type") == "text"
            and item.get("text", "").strip() == camvlm_prompts.build_qa_final_prompt()
        ):
            return True
    return False


def _has_complete_second_video_turn(example: Dict[str, Any]) -> bool:
    return any(
        _is_video_item(item) and _is_complete_second_video_item(item)
        for message in example.get("messages", [])
        for item in message.get("content", [])
    )


def prepare_dataset(example: Dict[str, Any]) -> Dict[str, Any]:
    raw_messages = copy.deepcopy(example["messages"])

    messages = camvlm_prompts.build_qa_initial_messages(example)
    message_index = 0
    while message_index < len(raw_messages):
        message = raw_messages[message_index]
        if message["role"] == "system":
            message_index += 1
            continue
        video_items = [
            item
            for item in message.get("content", [])
            if _is_video_item(item)
        ]
        if video_items and not all(_is_complete_second_video_item(item) for item in video_items):
            next_index = message_index + 1
            if next_index < len(raw_messages):
                next_message = raw_messages[next_index]
                if next_message.get("role") == "assistant" and not _is_final_answer_message(next_message):
                    message_index += 2
                    continue
            message_index += 1
            continue
        if _is_final_answer_message(message) and (
            not messages or not _is_final_answer_prompt_message(messages[-1])
        ):
            messages.append(_build_final_answer_prompt_message())
        if message["role"] == "assistant":
            for item in message["content"]:
                if item.get("type") == "text":
                    item["text"] = str(item["text"]).strip()
        messages.append(message)
        message_index += 1
    if not any(
        _is_video_item(item)
        for message in messages
        for item in message.get("content", [])
    ):
        raise ValueError(f"Sample {example.get('id')} has no complete 1-second video turns after pruning.")
    return {
        "messages": messages,
        "supervise_actions": bool(example.get("_supervise_actions", True)),
    }


def _mark_action_supervision(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    action_turns = 0
    answer_turns = 0
    for example in examples:
        if "options" in example or "problem" in example or "problem_type" in example:
            raise ValueError(
                "QA all-actions training requires the clean Question/Answer schema "
                "without options/problem/problem_type."
            )
        question = _get_question(example)
        answer = example.get("Answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Sample {example.get('id')} has an empty top-level Answer.")

        sample_answer_turns = 0
        for message in example.get("messages", []):
            if message.get("role") != "assistant":
                continue
            for item in message.get("content", []):
                if item.get("type") != "text":
                    continue
                text = str(item.get("text", "")).strip()
                if text.startswith("<answer>"):
                    expected = f"<answer>{answer.strip()}</answer>"
                    if text != expected:
                        raise ValueError(
                            f"Sample {example.get('id')} final answer does not match "
                            f"the top-level Answer."
                        )
                    sample_answer_turns += 1
                    answer_turns += 1
                elif '"action"' in text:
                    action_turns += 1
                else:
                    raise ValueError(
                        f"Sample {example.get('id')} has an unsupported assistant turn: "
                        f"{text[:120]}"
                    )
        if sample_answer_turns != 1:
            raise ValueError(
                f"Sample {example.get('id')} has {sample_answer_turns} final answer turns."
            )
        if not question:
            raise ValueError(f"Sample {example.get('id')} has an empty question.")
        example["_supervise_actions"] = True
    return {
        "samples": len(examples),
        "action_supervised_samples": len(examples),
        "answer_only_samples": 0,
        "action_turns": action_turns,
        "answer_turns": answer_turns,
    }


def _crop_video_to_window(
    video: torch.Tensor,
    window: Optional[List[float]],
) -> torch.Tensor:
    _, _, height, width = video.shape
    cropped = video
    if window and len(window) == 4:
        x1, y1, x2, y2 = window
        if not (x1 <= 0.0 and y1 <= 0.0 and x2 >= 1.0 and y2 >= 1.0):
            px1 = int(round(min(max(x1, 0.0), 1.0) * width))
            py1 = int(round(min(max(y1, 0.0), 1.0) * height))
            px2 = int(round(min(max(x2, 0.0), 1.0) * width))
            py2 = int(round(min(max(y2, 0.0), 1.0) * height))
            px1 = max(0, min(width - 1, px1))
            py1 = max(0, min(height - 1, py1))
            px2 = max(px1 + 1, min(width, px2))
            py2 = max(py1 + 1, min(height, py2))
            cropped = video[:, :, py1:py2, px1:px2]
    return cropped


def _video_tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    frames = video.detach().cpu().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _sample_absolute_indices(video_start: float, video_end: float, video_fps: float, total_frames: int) -> List[int]:
    duration = max(video_end - video_start, 0.0)
    requested_frames = max(2, int(math.floor(duration * TARGET_FPS + 1e-6)))
    if requested_frames % 2:
        requested_frames += 1
    nframes = min(requested_frames, MAX_FRAMES)
    end_time = max(video_start, video_end - 1.0 / max(video_fps, 1.0))
    if requested_frames > MAX_FRAMES:
        timestamps = np.linspace(video_start, end_time, num=nframes).tolist()
    else:
        timestamps = [min(video_start + frame_idx / TARGET_FPS, end_time) for frame_idx in range(nframes)]

    return [min(max(int(round(timestamp * video_fps)), 0), total_frames - 1) for timestamp in timestamps]


def _frame_sort_key(path: str) -> Tuple[int, Any]:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return 0, int(stem)
    except ValueError:
        return 1, stem


@lru_cache(maxsize=256)
def _list_frame_paths(video_dir: str) -> Tuple[str, ...]:
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"JPEG sequence directory not found: {video_dir}")
    frame_paths = [
        os.path.join(video_dir, name)
        for name in os.listdir(video_dir)
        if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
    ]
    frame_paths.sort(key=_frame_sort_key)
    if not frame_paths:
        raise ValueError(f"No image frames found in: {video_dir}")
    return tuple(frame_paths)


def _infer_source_fps(video_path: str) -> float:
    normalized = video_path.replace("\\", "/")
    if "/MeViS/" in normalized or normalized.startswith("MeViS/"):
        return DEFAULT_MEVIS_FPS
    if "/Refer-YouTube-VOS/" in normalized or normalized.startswith("Refer-YouTube-VOS/"):
        return DEFAULT_YOUTUBE_VOS_FPS
    raise ValueError(f"Could not infer source FPS for frame directory: {video_path}")


def _read_jpeg_frame(frame_path: str) -> torch.Tensor:
    with Image.open(frame_path) as image:
        frame = np.array(image.convert("RGB"))
    return torch.from_numpy(frame).permute(2, 0, 1).contiguous()


def _jpeg_frame_indices(ele: Dict[str, Any]) -> Tuple[Tuple[str, ...], List[int], float, int]:
    video_path = ele["video"]
    frame_paths = _list_frame_paths(video_path)
    total_frames = len(frame_paths)
    video_fps = float(ele.get("source_fps") or _infer_source_fps(video_path))
    video_start = float(ele.get("video_start", 0.0))
    video_end = float(ele.get("video_end", total_frames / video_fps))
    indices = _sample_absolute_indices(video_start, video_end, video_fps, total_frames)
    return frame_paths, indices, video_fps, total_frames


def _prefetch_jpeg_frames(video_items: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if not video_items:
        return {}

    frame_paths = []
    seen = set()
    for item in video_items:
        paths, indices, _, _ = _jpeg_frame_indices(item)
        for index in indices:
            frame_path = paths[index]
            if frame_path not in seen:
                seen.add(frame_path)
                frame_paths.append(frame_path)

    if not frame_paths:
        return {}
    if JPEG_READ_WORKERS <= 1 or len(frame_paths) == 1:
        return {frame_path: _read_jpeg_frame(frame_path) for frame_path in frame_paths}

    with ThreadPoolExecutor(max_workers=min(JPEG_READ_WORKERS, len(frame_paths))) as executor:
        frames = executor.map(_read_jpeg_frame, frame_paths)
        return dict(zip(frame_paths, frames))


def _fetch_video_jpeg(
    ele: Dict[str, Any],
    frame_cache: Dict[str, torch.Tensor],
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    video_path = ele["video"]
    frame_paths, indices, video_fps, total_frames = _jpeg_frame_indices(ele)

    frames = [frame_cache[frame_paths[index]] for index in indices]
    first_shape = frames[0].shape
    if any(frame.shape != first_shape for frame in frames[1:]):
        raise ValueError(f"Frames in JPEG sequence have inconsistent sizes: {video_path}")
    video = torch.stack(frames)
    video = _crop_video_to_window(video, ele.get("window"))
    metadata = {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": total_frames,
        "video_backend": "jpeg_sequence",
    }
    return _video_tensor_to_pil_frames(video), metadata


@lru_cache(maxsize=256)
def _mp4_metadata(video_path: str) -> Tuple[int, float]:
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError("decord is required by qwen-vl-utils to load MP4 training videos.") from exc
    reader = decord.VideoReader(video_path)
    return len(reader), float(reader.get_avg_fps())


def _mp4_time_bounds(ele: Dict[str, Any], video_fps: float, total_frames: int) -> Tuple[float, float]:
    duration = total_frames / video_fps
    start = max(0.0, min(float(ele.get("video_start", 0.0)), duration))
    end = max(0.0, min(float(ele.get("video_end", duration)), duration))
    if end <= start:
        end = min(duration, start + 1.0 / max(video_fps, 1.0))
    return start, end


def _fetch_video_decord(ele: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    import decord

    reader = decord.VideoReader(str(ele["video"]))
    total_frames = len(reader)
    video_fps = float(reader.get_avg_fps())
    if total_frames <= 0 or video_fps <= 0:
        raise ValueError(f"Invalid decord metadata for {ele['video']}: frames={total_frames}, fps={video_fps}")

    start, end = _mp4_time_bounds(ele, video_fps, total_frames)
    indices = _sample_absolute_indices(start, end, video_fps, total_frames)
    video = reader.get_batch(indices).asnumpy()
    video = torch.from_numpy(video).permute(0, 3, 1, 2).contiguous()
    metadata = {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": total_frames,
        "video_backend": "decord",
    }
    return video, metadata


def _fetch_video_qwen_backend(
    ele: Dict[str, Any],
) -> Tuple[List[Image.Image], Dict[str, Any]]:
    video, metadata = _fetch_video_decord(ele)
    video = _crop_video_to_window(video, ele.get("window"))
    return _video_tensor_to_pil_frames(video), metadata


def _build_qwen_video_item(
    item: Dict[str, Any],
    frames: List[Image.Image],
    metadata: Dict[str, Any],
    clip_total_pixels: Optional[int],
) -> Dict[str, Any]:
    qwen_item = {
        "type": "video",
        "video": frames,
        "sample_fps": TARGET_FPS,
        "raw_fps": float(metadata["fps"]),
        "min_pixels": MIN_TOKENS * QWEN3_VIDEO_PIXEL_SCALE,
        "_camvlm_video_metadata": metadata,
    }
    if clip_total_pixels is not None:
        qwen_item["total_pixels"] = clip_total_pixels
    return qwen_item


def _prepare_qwen_vision_messages(batch_messages: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    qwen_batch_messages = copy.deepcopy(batch_messages)
    for message_index, messages in enumerate(batch_messages):
        video_items = [
            item
            for message in messages
            for item in message.get("content", [])
            if _is_video_item(item)
        ]
        num_video_turns = len(video_items)
        clip_total_pixels = None
        if num_video_turns > 0 and TOTAL_TOKENS > 0:
            clip_total_pixels = int(TOTAL_TOKENS * QWEN3_VIDEO_PIXEL_SCALE / num_video_turns)
        jpeg_items = [item for item in video_items if os.path.isdir(item["video"])]
        jpeg_frame_cache = _prefetch_jpeg_frames(jpeg_items)
        for turn_index, message in enumerate(messages):
            for item_index, item in enumerate(message.get("content", [])):
                if _is_video_item(item):
                    if os.path.isdir(item["video"]):
                        frames, metadata = _fetch_video_jpeg(item, jpeg_frame_cache)
                    else:
                        frames, metadata = _fetch_video_qwen_backend(item)
                    qwen_batch_messages[message_index][turn_index]["content"][item_index] = _build_qwen_video_item(
                        item,
                        frames,
                        metadata,
                        clip_total_pixels,
                    )
    return qwen_batch_messages


def _collect_qwen_video_items(qwen_batch_messages: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    video_items = []
    unprepared_items = []
    for messages in qwen_batch_messages:
        for message in messages:
            for item in message.get("content", []):
                if item.get("type") != "video":
                    continue
                if "_camvlm_video_metadata" in item:
                    video_items.append(item)
                else:
                    unprepared_items.append({
                        "type": item.get("type"),
                        "video_type": type(item.get("video")).__name__,
                        "video": str(item.get("video"))[:200],
                    })
    if unprepared_items:
        raise ValueError(
            "Found video items that were not converted to qwen-vl-utils frame lists: "
            f"{json.dumps(unprepared_items[:5], ensure_ascii=False)}"
        )
    return video_items


def _process_qwen_video_items(
    qwen_video_items: List[Dict[str, Any]],
) -> Tuple[None, Optional[List[torch.Tensor]], Dict[str, Any]]:
    if not qwen_video_items:
        return None, None, {"do_sample_frames": False}

    video_inputs = []
    video_metadata = []
    for item in qwen_video_items:
        qwen_item = {key: value for key, value in item.items() if not key.startswith("_camvlm_")}
        if not isinstance(qwen_item.get("video"), (list, tuple)):
            raise ValueError(
                "Prepared qwen-vl-utils video item must contain a frame list, got "
                f"{type(qwen_item.get('video')).__name__}: {str(qwen_item.get('video'))[:200]}"
            )
        (video_input, _), _ = qwen_vision_process.fetch_video(
            qwen_item,
            image_patch_size=QWEN3_IMAGE_PATCH_SIZE,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
        video_inputs.append(video_input)
        video_metadata.append(item["_camvlm_video_metadata"])
    return None, video_inputs, {
        "do_sample_frames": False,
        "video_metadata": video_metadata,
    }


def _find_all_subsequences(sequence: List[int], subsequence: List[int]) -> List[int]:
    starts = []
    if not subsequence:
        return starts
    max_start = len(sequence) - len(subsequence)
    for start in range(max_start + 1):
        if sequence[start : start + len(subsequence)] == subsequence:
            starts.append(start)
    return starts


def _mask_labels(
    input_ids: torch.Tensor,
    supervise_actions: List[bool],
) -> torch.Tensor:
    if input_ids.size(0) != len(supervise_actions):
        raise ValueError(
            "supervise_actions must contain one flag per batch row: "
            f"rows={input_ids.size(0)} flags={len(supervise_actions)}."
        )

    labels = torch.full_like(input_ids, -100)
    message_start_ids = _token_ids("<|im_start|>")
    assistant_prefix_ids = _token_ids("<|im_start|>assistant\n")
    im_end_ids = _token_ids("<|im_end|>")
    answer_prefix_ids = _token_ids("<answer>")

    for row_idx, row in enumerate(input_ids):
        token_list = row.tolist()
        message_starts = _find_all_subsequences(token_list, message_start_ids)
        assistant_starts = _find_all_subsequences(token_list, assistant_prefix_ids)
        used_end_starts = set()

        for assistant_start in assistant_starts:
            content_start = assistant_start + len(assistant_prefix_ids)
            next_message_start = next(
                (start for start in message_starts if start > assistant_start),
                len(token_list),
            )
            end_starts = [
                content_start + offset
                for offset in _find_all_subsequences(
                    token_list[content_start:next_message_start],
                    im_end_ids,
                )
            ]
            if len(end_starts) != 1:
                raise ValueError("Could not find <|im_end|> after assistant response.")
            end_start = end_starts[0]
            if end_start in used_end_starts:
                raise ValueError("An <|im_end|> token was matched to multiple assistant turns.")
            used_end_starts.add(end_start)

            is_final_answer = (
                token_list[content_start : content_start + len(answer_prefix_ids)]
                == answer_prefix_ids
            )
            if supervise_actions[row_idx] or is_final_answer:
                span_end = end_start + len(im_end_ids)
                if span_end > next_message_start:
                    raise ValueError("Assistant supervision crossed into the next message.")
                labels[row_idx, content_start:span_end] = input_ids[
                    row_idx, content_start:span_end
                ]

        if len(used_end_starts) != len(assistant_starts):
            raise ValueError(
                "Assistant start/end markers are not one-to-one: "
                f"starts={len(assistant_starts)} ends={len(used_end_starts)}."
            )

    labels[labels == processor.tokenizer.pad_token_id] = -100
    for token in ("<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != processor.tokenizer.unk_token_id:
            labels[labels == token_id] = -100
    supervised_per_row = (labels != -100).sum(dim=1)
    if torch.any(supervised_per_row == 0):
        empty_rows = torch.nonzero(supervised_per_row == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"SFT batch rows contain no supervised tokens: {empty_rows}.")
    return labels


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    try:
        prepared = [prepare_dataset(example) for example in examples]
        batch_messages = [example["messages"] for example in prepared]
        supervise_actions = [example["supervise_actions"] for example in prepared]
        texts = [
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in batch_messages
        ]
        qwen_vision_messages = _prepare_qwen_vision_messages(batch_messages)
        image_inputs, video_inputs, processor_kwargs = _process_qwen_video_items(
            _collect_qwen_video_items(qwen_vision_messages)
        )

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            do_resize=False,
            **processor_kwargs,
        )
        inputs["labels"] = _mask_labels(inputs["input_ids"], supervise_actions)
        return inputs
    except AssertionError as exc:
        sample_context = [
            {
                "id": example.get("id"),
                "path": example.get("path"),
                "message_count": len(example.get("messages", [])),
            }
            for example in examples
        ]
        raise RuntimeError(
            "AssertionError while building a CamVLM SFT batch. "
            f"samples={json.dumps(sample_context, ensure_ascii=False)}\n"
            f"{traceback.format_exc()}"
        ) from exc


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, SFTFreezeConfig))
    script_args, training_args, model_config, freeze_config = parser.parse_args_and_config()

    training_args.freeze_llm = freeze_config.freeze_llm
    training_args.freeze_vision_tower = freeze_config.freeze_vision_tower
    training_args.freeze_merger = freeze_config.freeze_merger

    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    if script_args.dataset_name.endswith(".json"):
        with open(script_args.dataset_name, "r", encoding="utf-8") as f:
            rows = json.load(f)
        original_count = len(rows)
        rows = [row for row in rows if _has_complete_second_video_turn(row)]
        dropped_count = original_count - len(rows)
        if not rows:
            raise ValueError(
                f"No samples with at least one complete 1-second video turn in {script_args.dataset_name}."
            )
        if dropped_count:
            print(
                "Dropped SFT samples without a complete 1-second video turn: "
                f"{dropped_count}/{original_count}",
                flush=True,
            )
        supervision_stats = _mark_action_supervision(rows)
        print(f"Action supervision: {supervision_stats}", flush=True)
        dataset = DatasetDict({"train": Dataset.from_list(rows)})
    elif script_args.dataset_name.endswith(".jsonl"):
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        torch_dtype=torch_dtype,
        attn_implementation=getattr(model_config, "attn_implementation", None),
    )
    model_kwargs = {key: value for key, value in model_kwargs.items() if value is not None}

    model = _load_model(model_config.model_name_or_path, model_kwargs)
    model.config.use_cache = False

    model_to_configure = model
    compute_dtype = _resolve_compute_dtype(training_args)
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure, training_args, compute_dtype, training_args.device
    )

    if training_args.local_rank in (-1, 0):
        _print_trainable_parameters(model, freeze_config)

    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    processor.chat_template = QWEN3_CAMTRACK_CHAT_TEMPLATE

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=collate_fn,
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    _distributed_barrier_with_device()
    del model
    del trainer
    torch.cuda.empty_cache()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
