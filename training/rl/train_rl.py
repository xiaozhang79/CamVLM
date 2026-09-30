import os

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, HfArgumentParser

try:
    from transformers import AutoModelForImageTextToText, Qwen3VLForConditionalGeneration
except ImportError:
    AutoModelForImageTextToText = None
    Qwen3VLForConditionalGeneration = None

from .config import CamVLMGRPOArguments, DataArguments, ModelArguments
from .dataset import CamVLMRLDataset
from .prompts import QWEN3_CAMTRACK_CHAT_TEMPLATE
from .trainer import CamVLMInteractiveGRPOTrainer, preflight_vllm_support
from .vision import VisionLoader


def load_model(model_path: str, model_kwargs):
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **model_kwargs)
    if AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    return AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)


def set_requires_grad(parameters, requires_grad):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, training_args, compute_dtype, device):
    vision_tower = getattr(model, "visual", None)
    if vision_tower is None:
        return

    vision_tower.to(dtype=compute_dtype, device=device)

    vision_model_params = model.visual.parameters()
    set_requires_grad(vision_model_params, not training_args.freeze_vision_tower)

    merger = getattr(model.visual, "merger", None)
    if merger is not None:
        merger_params = merger.parameters()
        set_requires_grad(merger_params, not training_args.freeze_merger)


def configure_llm(model, training_args):
    lm_head = model.lm_head.parameters()
    set_requires_grad(lm_head, not training_args.freeze_llm)

    llm_params = model.model.parameters()
    set_requires_grad(llm_params, not training_args.freeze_llm)


def print_trainable_parameters(model):
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Trainable parameters: {trainable:,}/{total:,} ({100.0 * trainable / total:.2f}%)",
        flush=True,
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CamVLMGRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.num_generations != 8:
        raise ValueError("CamVLM GRPO is configured for exactly num_generations=8.")

    if training_args.bits != 16:
        raise ValueError("CamVLM RL uses full-precision weights with bits=16.")

    if training_args.use_vllm:
        installed_vllm = preflight_vllm_support(training_args.vllm_required_model_arch)
        if training_args.local_rank in (-1, 0):
            print(f"vLLM preflight passed: version={installed_vllm}", flush=True)

    # Save evaluable model/processor checkpoints without DeepSpeed optimizer
    # shards such as checkpoint-N/global_stepN.
    training_args.save_only_model = True
    training_args.remove_unused_columns = False
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    processor.chat_template = QWEN3_CAMTRACK_CHAT_TEMPLATE

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model = load_model(
        model_args.model_name_or_path,
        {
            "torch_dtype": compute_dtype,
            "attn_implementation": (
                "sdpa"
                if training_args.disable_flash_attn2
                else model_args.attn_implementation
            ),
            "trust_remote_code": model_args.trust_remote_code,
        },
    )

    model.config.use_cache = False

    model_to_configure = model
    configure_llm(model_to_configure, training_args)
    configure_vision_tower(
        model_to_configure,
        training_args,
        compute_dtype,
        training_args.device,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.local_rank in (-1, 0):
        print_trainable_parameters(model)

    dataset = CamVLMRLDataset(
        data_args.dataset_name,
        fps=data_args.fps,
    )

    if training_args.local_rank in (-1, 0):
        print(
            f"CamVLM GRPO dataset: samples={len(dataset)} sampler_seed={training_args.seed}",
            flush=True,
        )

    vision_loader = VisionLoader(
        fps=data_args.fps,
        min_tokens=data_args.min_tokens,
        total_tokens=data_args.total_tokens,
        max_frames=data_args.max_frames,
        mevis_fps=data_args.mevis_fps,
        youtube_vos_fps=data_args.youtube_vos_fps,
        jpeg_read_workers=max(1, int(os.environ.get("CAMTRACK_JPEG_READ_WORKERS", "8"))),
    )

    trainer = CamVLMInteractiveGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        vision_loader=vision_loader,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    trainer.save_state()

    model.config.use_cache = True

    if trainer.accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
        trainer.model.config.save_pretrained(training_args.output_dir)

    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
