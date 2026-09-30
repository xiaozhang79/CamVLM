import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser
from transformers import AutoModelForVision2Seq, AutoProcessor

PROJECT_SRC = Path(__file__).resolve().parents[2]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from evaluation.cctv_anomaly.task import TASK_PROMPT

try:
    from transformers import AutoModelForImageTextToText, Qwen3VLForConditionalGeneration
except ImportError:
    AutoModelForImageTextToText = None
    Qwen3VLForConditionalGeneration = None


TARGET_FPS = float(os.environ.get("CAMTRACK_SFT_FPS", "2"))
TOTAL_TOKENS = int(os.environ.get("CAMTRACK_TOTAL_TOKENS", "14336"))
MAX_FRAMES = int(os.environ.get("STAGE1_MAX_FRAMES", "448"))
MAX_SAMPLES = int(os.environ.get("STAGE1_MAX_SAMPLES", "0"))
MIN_TOKENS = int(os.environ.get("STAGE1_MIN_TOKENS", "64"))
QWEN3_IMAGE_PATCH_SIZE = 16
QWEN3_VIDEO_PIXEL_SCALE = 32**2
processor = None

REQUIRED_FIELDS = ("video_id", "duration", "video_path", "caption", "Category")
VALID_CATEGORIES = {
    "break-in",
    "theft",
    "suspicious behavior",
    "violence",
    "vandalism",
    "fire hazard",
    "personal emergency",
    "wild animal",
    "vehicle incident",
    "normal activity",
}

@dataclass
class SFTFreezeConfig:
    freeze_llm: bool = field(default=False)
    freeze_vision_tower: bool = field(default=True)
    freeze_merger: bool = field(default=False)


def _load_model(model_name_or_path: str, model_kwargs: Dict[str, Any]):
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_name_or_path, **model_kwargs)
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)
    return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **model_kwargs)


def _set_requires_grad(parameters, requires_grad: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def _get_visual_tower(model):
    visual = getattr(model, "visual", None)
    if visual is not None:
        return visual
    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is not None:
        return visual
    raise ValueError("Could not locate the visual tower.")


def _configure_model(model, freeze_config: SFTFreezeConfig, compute_dtype, device) -> None:
    visual = _get_visual_tower(model)
    visual.to(dtype=compute_dtype, device=device)
    _set_requires_grad(visual.parameters(), not freeze_config.freeze_vision_tower)
    merger = getattr(visual, "merger", None)
    if merger is None:
        raise ValueError("Could not locate the visual merger.")
    _set_requires_grad(merger.parameters(), not freeze_config.freeze_merger)

    lm_head = getattr(model, "lm_head", None)
    backbone = getattr(model, "model", None)
    llm = getattr(backbone, "language_model", None)
    if lm_head is None or llm is None:
        raise ValueError("Could not locate the language model or lm_head.")
    _set_requires_grad(lm_head.parameters(), not freeze_config.freeze_llm)
    _set_requires_grad(llm.parameters(), not freeze_config.freeze_llm)


def _resolve_compute_dtype(training_args) -> torch.dtype:
    if training_args.fp16:
        return torch.float16
    if training_args.bf16:
        return torch.bfloat16
    return torch.float32


def _print_trainable_parameters(model, freeze_config: SFTFreezeConfig) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if total == 0:
        print(
            "Trainable parameter count is unavailable before trainer initialization; "
            f"freeze_llm={freeze_config.freeze_llm}, "
            f"freeze_vision_tower={freeze_config.freeze_vision_tower}, "
            f"freeze_merger={freeze_config.freeze_merger}",
            flush=True,
        )
        return
    print(
        f"Trainable parameters: {trainable:,}/{total:,} ({100.0 * trainable / total:.2f}%); "
        f"freeze_llm={freeze_config.freeze_llm}, "
        f"freeze_vision_tower={freeze_config.freeze_vision_tower}, "
        f"freeze_merger={freeze_config.freeze_merger}",
        flush=True,
    )


def _build_messages(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": example["video_path"],
                    "min_pixels": int(MIN_TOKENS * QWEN3_VIDEO_PIXEL_SCALE),
                    "total_pixels": int(TOTAL_TOKENS * QWEN3_VIDEO_PIXEL_SCALE),
                    "fps": TARGET_FPS,
                    "max_frames": MAX_FRAMES,
                },
                {"type": "text", "text": TASK_PROMPT},
            ],
        }
    ]


def _append_answer(messages: List[Dict[str, Any]], example: Dict[str, Any]) -> None:
    messages.append(
        {
            "role": "assistant",
            "content": (
                f"Video category: {example['Category']}\n"
                f"Video description: {example['caption']}"
            ),
        }
    )


def _token_ids(text: str) -> List[int]:
    return processor.tokenizer(text, add_special_tokens=False)["input_ids"]


def _find_all_subsequences(sequence: List[int], subsequence: List[int]) -> List[int]:
    if not subsequence:
        return []
    return [
        start
        for start in range(len(sequence) - len(subsequence) + 1)
        if sequence[start : start + len(subsequence)] == subsequence
    ]


def _mask_labels(input_ids: torch.Tensor) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    message_start_ids = _token_ids("<|im_start|>")
    assistant_prefix_ids = _token_ids("<|im_start|>assistant\n")
    im_end_ids = _token_ids("<|im_end|>")
    for row_index, row in enumerate(input_ids):
        token_list = row.tolist()
        message_starts = _find_all_subsequences(token_list, message_start_ids)
        assistant_starts = _find_all_subsequences(token_list, assistant_prefix_ids)
        if len(assistant_starts) != 1:
            raise ValueError(f"Expected one assistant answer, found {len(assistant_starts)}.")
        content_start = assistant_starts[0] + len(assistant_prefix_ids)
        next_message_start = next((start for start in message_starts if start > assistant_starts[0]), len(token_list))
        end_starts = [
            content_start + offset
            for offset in _find_all_subsequences(token_list[content_start:next_message_start], im_end_ids)
        ]
        if len(end_starts) != 1:
            raise ValueError("Could not find the assistant message end token.")
        span_end = end_starts[0] + len(im_end_ids)
        labels[row_index, content_start:span_end] = input_ids[row_index, content_start:span_end]
    labels[labels == processor.tokenizer.pad_token_id] = -100
    for token in ("<|vision_start|>", "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"):
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id is not None and token_id != processor.tokenizer.unk_token_id:
            labels[labels == token_id] = -100
    if torch.any((labels != -100).sum(dim=1) == 0):
        raise ValueError("Stage 1 batch contains a row with no supervised answer tokens.")
    return labels


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    try:
        messages = [_build_messages(example) for example in examples]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=QWEN3_IMAGE_PATCH_SIZE,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is None or len(video_inputs) != len(examples):
            raise ValueError("Stage 1 batch has empty or incomplete video inputs.")
        video_inputs, video_metadatas = zip(*video_inputs)
        for message, example in zip(messages, examples):
            _append_answer(message, example)
        texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=list(video_inputs),
            video_metadata=list(video_metadatas),
            return_tensors="pt",
            padding=True,
            do_resize=False,
            **video_kwargs,
        )
        inputs["labels"] = _mask_labels(inputs["input_ids"])
        return inputs
    except Exception as exc:
        context = [{"video_id": example.get("video_id"), "video_path": example.get("video_path")} for example in examples]
        raise RuntimeError(
            f"Failed to build Stage 1 SFT batch: {context}\n{traceback.format_exc()}"
        ) from exc


def _validate_dataset(dataset: Dataset) -> None:
    if MAX_FRAMES < 2:
        raise ValueError("STAGE1_MAX_FRAMES must be at least 2.")
    if TARGET_FPS <= 0.0:
        raise ValueError("CAMTRACK_SFT_FPS must be positive.")
    if TOTAL_TOKENS <= 0:
        raise ValueError("CAMTRACK_TOTAL_TOKENS must be positive.")
    if MIN_TOKENS <= 0:
        raise ValueError("STAGE1_MIN_TOKENS must be positive.")
    if tuple(dataset.column_names) != REQUIRED_FIELDS:
        raise ValueError(f"Stage 1 dataset fields must be exactly {REQUIRED_FIELDS}, got {dataset.column_names}.")
    for example in dataset:
        if not isinstance(example["video_id"], str) or not example["video_id"]:
            raise ValueError("Stage 1 dataset contains an empty video_id.")
        if not isinstance(example["video_path"], str) or not Path(example["video_path"]).is_file():
            raise ValueError(f"Missing Stage 1 video: {example['video_path']!r}")
        if not isinstance(example["caption"], str) or not example["caption"].strip():
            raise ValueError(f"Stage 1 sample {example['video_id']} has an empty caption.")
        if example["Category"] not in VALID_CATEGORIES:
            raise ValueError(f"Stage 1 sample {example['video_id']} has invalid Category: {example['Category']!r}")
        try:
            if float(example["duration"]) <= 0.0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Stage 1 sample {example['video_id']} has invalid duration: {example['duration']!r}") from exc


def _distributed_barrier_with_device() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        else:
            torch.distributed.barrier()


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, SFTFreezeConfig))
    script_args, training_args, model_config, freeze_config = parser.parse_args_and_config()
    if not script_args.dataset_name.endswith(".jsonl"):
        raise ValueError("Stage 1 requires a local JSONL dataset.")

    dataset = Dataset.from_json(script_args.dataset_name)
    _validate_dataset(dataset)
    if MAX_SAMPLES > 0:
        dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))
    print(
        f"Stage 1 dataset: samples={len(dataset)}, fps={TARGET_FPS}, max_frames={MAX_FRAMES}, "
        f"min_tokens={MIN_TOKENS}, total_tokens={TOTAL_TOKENS}",
        flush=True,
    )

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.freeze_llm = freeze_config.freeze_llm
    training_args.freeze_vision_tower = freeze_config.freeze_vision_tower
    training_args.freeze_merger = freeze_config.freeze_merger

    torch_dtype = model_config.torch_dtype if model_config.torch_dtype in ["auto", None] else getattr(torch, model_config.torch_dtype)
    model_kwargs = {
        key: value
        for key, value in {
            "revision": model_config.model_revision,
            "trust_remote_code": model_config.trust_remote_code,
            "torch_dtype": torch_dtype,
            "attn_implementation": getattr(model_config, "attn_implementation", None),
        }.items()
        if value is not None
    }
    model = _load_model(model_config.model_name_or_path, model_kwargs)
    model.config.use_cache = False
    _configure_model(model, freeze_config, _resolve_compute_dtype(training_args), training_args.device)
    if training_args.local_rank in (-1, 0):
        _print_trainable_parameters(model, freeze_config)

    processor = AutoProcessor.from_pretrained(model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
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
