import importlib.metadata
import copy
import os
from collections import defaultdict
from collections.abc import Sized
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from accelerate.utils import gather, gather_object, set_seed
from torch.utils.data import DataLoader, Sampler
from transformers import GenerationConfig, Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker
from trl.trainer.utils import disable_dropout_in_model, selective_log_softmax

from training.rl.window_utils import parse_initial_window

from .prompts import QWEN3_CAMTRACK_CHAT_TEMPLATE, build_final_prompt, build_initial_messages
from .rewards import (
    REWARD_WEIGHTS,
    ParsedAction,
    compute_trajectory_reward,
    parse_action_output,
    parse_answer_letter,
    replay_action,
)
from .vision import VisionLoader


REWARD_METADATA_KEYS = {
    "target_object_ids",
    "target_bboxes",
    "target_bbox_fallbacks",
}


class RepeatSampler(Sampler):
    """Repeat each prompt across ranks so one distributed group forms one GRPO group."""

    def __init__(
        self,
        data_source: Sized,
        *,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            indices = torch.randperm(len(self.data_source), generator=self.generator).tolist()
        else:
            indices = list(range(len(self.data_source)))
        chunks = [
            indices[start : start + self.batch_size]
            for start in range(0, len(indices), self.batch_size)
        ]
        for chunk in chunks:
            if len(chunk) != self.batch_size:
                continue
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        usable = len(self.data_source) // self.batch_size * self.batch_size
        return usable * self.mini_repeat_count * self.repeat_count


@contextmanager
def unwrap_model_for_generation(
    model,
    accelerator,
    gather_deepspeed3_params: bool = True,
):
    unwrapped = accelerator.unwrap_model(model)
    gradient_checkpointing = getattr(unwrapped, "is_gradient_checkpointing", False)
    if gradient_checkpointing:
        unwrapped.gradient_checkpointing_disable()
    plugin = accelerator.state.deepspeed_plugin
    if plugin is not None and plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
            if gradient_checkpointing:
                unwrapped.gradient_checkpointing_enable()
            return
        import deepspeed

        with deepspeed.zero.GatheredParameters(model.parameters()):
            yield accelerator.unwrap_model(model)
    else:
        yield unwrapped
    if gradient_checkpointing:
        unwrapped.gradient_checkpointing_enable()


def _identity_collator(features):
    return features


def _find_subsequences(sequence: List[int], subsequence: List[int]) -> List[int]:
    if not subsequence:
        return []
    return [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if sequence[index : index + len(subsequence)] == subsequence
    ]


def _load_vllm_model_weights(model, weights):
    if not hasattr(model, "load_weights"):
        raise RuntimeError(
            f"vLLM model {type(model).__name__} does not expose load_weights()."
        )
    return model.load_weights(weights)


def preflight_vllm_support(required_arch: str) -> str:
    try:
        installed_version = importlib.metadata.version("vllm")
        import vllm
        from vllm.model_executor.models import ModelRegistry
    except Exception as exc:
        raise RuntimeError("vLLM mode requires an importable vllm installation.") from exc
    try:
        supported_archs = set(ModelRegistry.get_supported_archs())
    except Exception:
        registry_path = Path(vllm.__file__).resolve().parent / "model_executor/models/registry.py"
        registry_text = registry_path.read_text(encoding="utf-8") if registry_path.exists() else ""
        supported_archs = {required_arch} if required_arch in registry_text else set()
    if required_arch not in supported_archs:
        raise RuntimeError(
            f"vLLM {installed_version} does not register {required_arch}. "
            "Install a vLLM release with Qwen3-VL multimodal support before using --use_vllm True."
        )
    return installed_version


def compute_bnpo_policy_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: Optional[torch.Tensor],
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon_low: float,
    epsilon_high: float,
    delta: Optional[float] = None,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if old_per_token_logps is None:
        old_per_token_logps = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    if delta is not None:
        coef_1 = torch.clamp(coef_1, max=delta)

    if advantages.ndim == 1:
        token_advantages = advantages.unsqueeze(1)
    elif advantages.shape == per_token_logps.shape:
        token_advantages = advantages
    else:
        raise ValueError(
            "advantages must be one scalar per sequence or one value per "
            f"generated token, got {tuple(advantages.shape)} for "
            f"log-probs {tuple(per_token_logps.shape)}."
        )
    per_token_loss1 = coef_1 * token_advantages
    per_token_loss2 = coef_2 * token_advantages
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    denominator = completion_mask.sum().clamp(min=1.0)
    loss = (per_token_loss * completion_mask).sum() / denominator

    is_low_clipped = (coef_1 < 1 - epsilon_low) & (token_advantages < 0)
    is_high_clipped = (coef_1 > 1 + epsilon_high) & (token_advantages > 0)
    metrics = {
        "clip_ratio/low_mean": (is_low_clipped * completion_mask).sum() / denominator,
        "clip_ratio/high_mean": (is_high_clipped * completion_mask).sum() / denominator,
        "clip_ratio/region_mean": (
            ((is_low_clipped | is_high_clipped) * completion_mask).sum() / denominator
        ),
    }
    return loss, metrics


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: List[int]
    generated: bool
    terminated_with_eos: bool
    truncated: bool
    canonicalized: bool = False


def should_supervise_generation(
    result: GenerationResult,
    *,
    mask_truncated_completions: bool,
) -> bool:
    return result.generated and not (
        mask_truncated_completions and result.truncated
    )


def policy_target_and_logit_positions(
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.size(0) != 1:
        raise ValueError("Interactive CamVLM GRPO requires one trajectory per rank.")
    target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if target_positions.numel() == 0:
        raise ValueError("The rollout contains no generated tokens eligible for policy loss.")
    if torch.any(target_positions == 0):
        raise ValueError("A supervised token cannot appear at sequence position zero.")
    return target_positions, target_positions - 1


class CamVLMInteractiveGRPOTrainer(Trainer):
    def __init__(self, *args, vision_loader: VisionLoader, **kwargs):
        kwargs["data_collator"] = _identity_collator
        kwargs["model"].warnings_issued["estimate_tokens"] = True
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False
        self.vision_loader = vision_loader
        self.processor = self.processing_class
        self.processor.chat_template = QWEN3_CAMTRACK_CHAT_TEMPLATE
        self.num_generations = self.args.num_generations
        self.num_iterations = self.args.num_iterations
        self.epsilon_low = self.args.epsilon
        self.epsilon_high = (
            self.args.epsilon_high
            if self.args.epsilon_high is not None
            else self.args.epsilon
        )
        self.repetition_penalty = self.args.repetition_penalty
        self.min_p = self.args.min_p
        self.shuffle_dataset = self.args.shuffle_dataset
        self.initial_window = parse_initial_window(self.args.initial_window)
        self.reward_func_names = [
            name.strip()
            for name in self.args.reward_funcs.split(",")
            if name.strip()
        ]
        unknown_rewards = set(self.reward_func_names) - set(REWARD_WEIGHTS)
        if unknown_rewards:
            raise ValueError(
                f"Unknown reward functions: {sorted(unknown_rewards)}. "
                f"Available rewards: {sorted(REWARD_WEIGHTS)}"
            )
        if not self.reward_func_names:
            raise ValueError("At least one reward function must be enabled.")
        if "accuracy" in self.reward_func_names:
            auxiliary_weight = sum(
                REWARD_WEIGHTS[name]
                for name in self.reward_func_names
                if name != "accuracy"
            )
            if REWARD_WEIGHTS["accuracy"] <= auxiliary_weight:
                raise ValueError(
                    "Accuracy reward must be greater than the sum of enabled "
                    "auxiliary reward weights so a wrong answer can never outrank "
                    "a correct answer. "
                    f"accuracy={REWARD_WEIGHTS['accuracy']} auxiliary={auxiliary_weight}"
                )
        self._reward_metrics: Dict[str, List[float]] = defaultdict(list)
        self._textual_logs: List[Dict[str, Any]] = []
        self._last_vllm_sync_step = -1

        if self.args.per_device_train_batch_size != 1:
            raise ValueError("Interactive CamVLM GRPO currently requires per_device_train_batch_size=1.")
        if self.accelerator.num_processes != self.num_generations:
            raise ValueError(
                "This trainer maps one generation to each rank. "
                f"world_size={self.accelerator.num_processes} must equal num_generations={self.num_generations}."
            )
        if self.args.steps_per_generation != 1:
            raise ValueError("Interactive CamVLM GRPO currently requires steps_per_generation=1.")
        if self.args.beta != 0.0:
            raise ValueError("This implementation intentionally supports beta=0 only; no reference model is loaded.")
        if self.args.loss_type != "bnpo":
            raise ValueError("This implementation currently supports loss_type=bnpo only.")
        if self.args.max_final_new_tokens >= self.args.max_completion_length:
            raise ValueError(
                "max_final_new_tokens must be smaller than max_completion_length "
                "so interactive action turns retain a nonzero token budget. "
                f"Got final={self.args.max_final_new_tokens}, "
                f"completion={self.args.max_completion_length}."
            )
        if self.args.max_action_new_tokens <= 0:
            raise ValueError("max_action_new_tokens must be greater than zero.")
        if self.args.use_vllm and self.args.vllm_mode != "colocate":
            raise ValueError("CamVLM's eight-instance vLLM backend requires vllm_mode=colocate.")
        if self.args.disable_dropout:
            disable_dropout_in_model(self.model)

        set_seed(self.args.seed, device_specific=True)
        generation_kwargs = {
            "do_sample": True,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "top_k": self.args.top_k,
            "min_p": self.min_p,
            "repetition_penalty": self.repetition_penalty,
            "cache_implementation": self.args.cache_implementation,
            "pad_token_id": self.processor.tokenizer.pad_token_id,
            "eos_token_id": self.processor.tokenizer.eos_token_id,
        }
        if self.args.generation_kwargs is not None:
            generation_kwargs.update(self.args.generation_kwargs)
        action_generation_kwargs = dict(generation_kwargs)
        action_generation_kwargs["max_new_tokens"] = self.args.max_action_new_tokens
        final_generation_kwargs = dict(generation_kwargs)
        final_generation_kwargs["max_new_tokens"] = self.args.max_final_new_tokens
        self.action_generation_config = GenerationConfig(**action_generation_kwargs)
        self.final_generation_config = GenerationConfig(**final_generation_kwargs)
        self.llm = None
        if self.args.use_vllm:
            self._init_vllm()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["id", "Question", "options", "Answer", "video_items"]

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        global_generation_batch = (
            self.args.per_device_train_batch_size
            * self.accelerator.num_processes
            * self.args.steps_per_generation
        )
        unique_prompts = global_generation_batch // self.num_generations
        return RepeatSampler(
            dataset,
            mini_repeat_count=self.num_generations,
            batch_size=unique_prompts,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Training requires a dataset.")
        params = {
            "batch_size": self._train_batch_size,
            "sampler": self._get_train_sampler(),
            "collate_fn": self.data_collator,
            "drop_last": self.args.dataloader_drop_last,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_num_workers > 0:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
            params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )
        return self.accelerator.prepare(DataLoader(self.train_dataset, **params))

    def _vllm_preflight(self):
        preflight_vllm_support(self.args.vllm_required_model_arch)

    def _init_vllm(self):
        self._vllm_preflight()
        from vllm import LLM

        self.llm = LLM(
            model=self.model.name_or_path,
            tensor_parallel_size=self.args.vllm_tensor_parallel_size,
            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
            max_model_len=self.args.vllm_max_model_len,
            max_num_seqs=(
                self.args.per_device_train_batch_size
                * self.args.vllm_tensor_parallel_size
                * self.args.gradient_accumulation_steps
            ),
            max_num_batched_tokens=4096,
            distributed_executor_backend="external_launcher",
            seed=self.accelerator.process_index,
            limit_mm_per_prompt={"video": self.args.vllm_max_video_clips},
            trust_remote_code=True,
        )
        self.accelerator.wait_for_everyone()

    def _sync_model_to_vllm(self):
        if self.llm is None:
            return
        if self._last_vllm_sync_step == self.state.global_step:
            return
        source = self.accelerator.unwrap_model(self.model_wrapped)
        weights = [
            (name, parameter.data)
            for name, parameter in source.named_parameters()
        ]
        if hasattr(self.llm, "apply_model"):
            self.llm.apply_model(
                partial(_load_vllm_model_weights, weights=weights)
            )
            if hasattr(self.llm, "reset_prefix_cache"):
                self.llm.reset_prefix_cache()
            self._last_vllm_sync_step = self.state.global_step
            return

        executor = self.llm.llm_engine.model_executor
        driver_worker = getattr(executor, "driver_worker", None)
        if driver_worker is None:
            raise RuntimeError("Unsupported vLLM executor: could not access driver_worker for weight sync.")
        runner = driver_worker.model_runner
        vllm_model = getattr(runner, "model", None)
        if vllm_model is None or not hasattr(vllm_model, "load_weights"):
            raise RuntimeError("Unsupported vLLM model runner: load_weights is unavailable.")
        vllm_model.load_weights(weights)
        if hasattr(self.llm, "reset_prefix_cache"):
            self.llm.reset_prefix_cache()
        self._last_vllm_sync_step = self.state.global_step

    def _video_items(self, messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            item
            for message in messages
            for item in message.get("content", [])
            if self.vision_loader.is_video_item(item)
        ]

    def _processor_inputs(
        self,
        messages,
        *,
        add_generation_prompt: bool,
        budget_video_turns: Optional[int] = None,
    ):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        _, videos, metadata = self.vision_loader.collect_from_messages(
            messages,
            budget_video_turns=budget_video_turns,
        )
        kwargs: Dict[str, Any] = {"do_sample_frames": False}
        if metadata is not None:
            kwargs["video_metadata"] = metadata
        inputs = self.processor(
            text=[text],
            videos=videos,
            return_tensors="pt",
            padding=True,
            **kwargs,
        )
        return inputs

    def _generation_result_from_ids(
        self,
        token_ids: Sequence[int],
        *,
        max_new_tokens: int,
        terminated_with_eos: Optional[bool] = None,
    ) -> GenerationResult:
        ids = list(map(int, token_ids))
        raw_length = len(ids)
        eos_id = self.processor.tokenizer.eos_token_id
        eos_in_ids = eos_id in ids
        terminated = eos_in_ids if terminated_with_eos is None else terminated_with_eos
        if eos_in_ids:
            ids = ids[: ids.index(eos_id)]
        text = self.processor.tokenizer.decode(
            ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        roundtrip = self.processor.tokenizer(
            text,
            add_special_tokens=False,
        )["input_ids"]
        # BPE tokenization is not injective: multiple valid token sequences can
        # decode to the same text, while encoding that text chooses one
        # canonical segmentation. The complete interactive dialogue is rebuilt
        # from text before the policy forward pass, so supervise that canonical
        # segmentation instead of rejecting an otherwise valid generation.
        canonicalized = roundtrip != ids
        ids = list(map(int, roundtrip))
        return GenerationResult(
            text=text,
            token_ids=ids + ([eos_id] if terminated else []),
            generated=True,
            terminated_with_eos=terminated,
            truncated=(not terminated and raw_length >= max_new_tokens),
            canonicalized=canonicalized,
        )

    def _generate_transformers(
        self,
        messages,
        generation_config,
        *,
        budget_video_turns: int,
    ) -> GenerationResult:
        inputs = self._processor_inputs(
            messages,
            add_generation_prompt=True,
            budget_video_turns=budget_video_turns,
        )
        inputs = {key: value.to(self.accelerator.device) if torch.is_tensor(value) else value for key, value in inputs.items()}
        with unwrap_model_for_generation(
            self.model_wrapped,
            self.accelerator,
            gather_deepspeed3_params=self.args.ds3_gather_for_generation,
        ) as model, torch.no_grad():
            generated = model.generate(
                **inputs,
                generation_config=generation_config,
                use_model_defaults=False,
            )
        completion_ids = generated[0, inputs["input_ids"].shape[1] :].tolist()
        return self._generation_result_from_ids(
            completion_ids,
            max_new_tokens=generation_config.max_new_tokens,
        )

    def _generate_vllm(
        self,
        messages,
        generation_config,
        *,
        budget_video_turns: int,
    ) -> GenerationResult:
        from vllm import SamplingParams

        self._sync_model_to_vllm()
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        multimodal = self.vision_loader.vllm_multimodal_data(
            messages,
            budget_video_turns=budget_video_turns,
        )
        sampling_kwargs = {
            "n": 1,
            "max_tokens": generation_config.max_new_tokens,
            "temperature": generation_config.temperature,
            "top_p": self.args.top_p,
            "top_k": -1 if self.args.top_k is None else self.args.top_k,
            "min_p": 0.0 if self.min_p is None else self.min_p,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.args.generation_kwargs is not None:
            sampling_kwargs.update(self.args.generation_kwargs)
        sampling_kwargs["max_tokens"] = generation_config.max_new_tokens
        sampling = SamplingParams(
            **sampling_kwargs,
        )
        outputs = self.llm.generate(
            [{"prompt": prompt, "multi_modal_data": multimodal}],
            sampling_params=sampling,
            use_tqdm=False,
        )
        output = outputs[0].outputs[0]
        return self._generation_result_from_ids(
            output.token_ids,
            max_new_tokens=generation_config.max_new_tokens,
            terminated_with_eos=getattr(output, "finish_reason", None) == "stop",
        )

    def _generate(
        self,
        messages,
        generation_config,
        *,
        budget_video_turns: int,
    ) -> GenerationResult:
        if self.args.use_vllm:
            return self._generate_vllm(
                messages,
                generation_config,
                budget_video_turns=budget_video_turns,
            )
        return self._generate_transformers(
            messages,
            generation_config,
            budget_video_turns=budget_video_turns,
        )

    @staticmethod
    def _generation_config_with_limit(
        generation_config: GenerationConfig,
        max_new_tokens: int,
    ) -> GenerationConfig:
        limited = copy.deepcopy(generation_config)
        limited.max_new_tokens = max_new_tokens
        return limited

    def _rollout(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        messages = build_initial_messages(sample)
        gt_items = sample["video_items"]
        target_bboxes_by_turn = [item["target_bboxes"] for item in gt_items]
        target_bbox_fallbacks_by_turn = [item["target_bbox_fallbacks"] for item in gt_items]
        current_window = self.initial_window
        parsed_actions: List[ParsedAction] = []
        predicted_after: List[List[float]] = []
        action_outputs: List[str] = []
        generation_results: List[GenerationResult] = []
        generated_tokens = 0
        action_token_budget = max(
            0,
            self.args.max_completion_length - self.args.max_final_new_tokens,
        )

        for gt_item in gt_items:
            current_item = {
                key: value
                for key, value in gt_item.items()
                if key not in REWARD_METADATA_KEYS
            }
            current_item["window"] = [round(value, 4) for value in current_window]
            messages.append({"role": "user", "content": [current_item]})
            if generated_tokens >= action_token_budget:
                action_result = GenerationResult(
                    text='{"action": None}',
                    token_ids=[],
                    generated=False,
                    terminated_with_eos=False,
                    truncated=False,
                )
            else:
                remaining_action_tokens = action_token_budget - generated_tokens
                action_result = self._generate(
                    messages,
                    self._generation_config_with_limit(
                        self.action_generation_config,
                        min(
                            self.args.max_action_new_tokens,
                            remaining_action_tokens,
                        ),
                    ),
                    budget_video_turns=len(gt_items),
                )
            action_output = action_result.text
            generated_tokens += len(action_result.token_ids)
            parsed = parse_action_output(action_output)
            current_window = replay_action(current_window, parsed)
            parsed_actions.append(parsed)
            predicted_after.append(list(current_window))
            action_outputs.append(action_output)
            generation_results.append(action_result)
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": action_output}],
            })

        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": build_final_prompt(len(sample["options"]))}],
        })
        remaining_completion_tokens = self.args.max_completion_length - generated_tokens
        if remaining_completion_tokens > 0:
            final_result = self._generate(
                messages,
                self._generation_config_with_limit(
                    self.final_generation_config,
                    min(
                        self.args.max_final_new_tokens,
                        remaining_completion_tokens,
                    ),
                ),
                budget_video_turns=len(gt_items),
            )
        else:
            final_result = GenerationResult(
                text="",
                token_ids=[],
                generated=False,
                terminated_with_eos=False,
                truncated=False,
            )
        final_output = final_result.text
        generation_results.append(final_result)
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": final_output}],
        })
        reward = compute_trajectory_reward(
            predicted_after,
            target_bboxes_by_turn,
            target_bbox_fallbacks_by_turn,
            parsed_actions,
            final_output,
            sample["Answer"],
            len(sample["options"]),
            reward_funcs=self.reward_func_names,
        )
        return {
            "messages": messages,
            "reward": reward,
            "action_outputs": action_outputs,
            "final_output": final_output,
            "generation_results": generation_results,
        }

    def _assistant_labels(
        self,
        input_ids: torch.Tensor,
        generation_results: Sequence[GenerationResult],
    ) -> torch.Tensor:
        labels = torch.full_like(input_ids, -100)
        prefix = self.processor.tokenizer(
            "<|im_start|>assistant\n",
            add_special_tokens=False,
        )["input_ids"]
        ending = self.processor.tokenizer("<|im_end|>", add_special_tokens=False)["input_ids"]
        for row_index, row in enumerate(input_ids):
            values = row.tolist()
            assistant_starts = _find_subsequences(values, prefix)
            if len(assistant_starts) != len(generation_results):
                raise ValueError(
                    f"Expected {len(generation_results)} assistant turns, found {len(assistant_starts)}."
                )
            for result_index, (start, result) in enumerate(
                zip(assistant_starts, generation_results)
            ):
                content_start = start + len(prefix)
                end_offsets = _find_subsequences(values[content_start:], ending)
                if not end_offsets:
                    raise ValueError("Generated assistant turn has no closing <|im_end|> token.")
                content_end = content_start + end_offsets[0]
                if not should_supervise_generation(
                    result,
                    mask_truncated_completions=self.args.mask_truncated_completions,
                ):
                    continue
                supervised_end = content_end + (len(ending) if result.terminated_with_eos else 0)
                actual_ids = values[content_start:supervised_end]
                if actual_ids != result.token_ids:
                    raise RuntimeError(
                        "The rebuilt assistant span does not match its canonical "
                        "generated token IDs; refusing to compute policy loss on "
                        "a different completion."
                    )
                labels[row_index, content_start:supervised_end] = row[content_start:supervised_end]
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return labels

    def _record_reward_metrics(self, reward, group_rewards: torch.Tensor):
        group_std = group_rewards.float().std(unbiased=True).item()
        values = {
            "rewards/answer_accuracy": reward.answer_accuracy,
            "rewards/viewpoint": reward.mean_viewpoint,
            "rewards/total": reward.total_reward,
            "accuracy": reward.answer_accuracy,
            "mean_viewpoint": reward.mean_viewpoint,
            "rewards/bbox_transition_count": reward.bbox_transition_count,
            "rewards/bbox_fallback_transition_count": reward.bbox_fallback_transition_count,
            "mean_action_seconds": reward.action_seconds,
            "rollout/action_seconds": reward.action_seconds,
            "rollout/invalid_action_rate": (
                reward.invalid_action_seconds / reward.decision_seconds
                if reward.decision_seconds
                else 0.0
            ),
            "invalid_output_rate": (
                (reward.invalid_action_seconds + int(not reward.final_answer_valid))
                / (reward.decision_seconds + 1)
            ),
            "reward_std": group_std,
            "fraction_zero_std": float(group_std < 1e-8),
        }
        for key, value in values.items():
            gathered_value = gather(torch.tensor(float(value), device=self.accelerator.device)).mean().item()
            self._reward_metrics[key].append(gathered_value)

    def _record_completion_metrics(self, generation_results: Sequence[GenerationResult]):
        generated_results = [result for result in generation_results if result.generated]
        local_length = torch.tensor(
            float(sum(len(result.token_ids) for result in generated_results)),
            device=self.accelerator.device,
        )
        local_truncated = torch.tensor(
            float(any(result.truncated for result in generated_results)),
            device=self.accelerator.device,
        )
        local_canonicalized = torch.tensor(
            (
                sum(result.canonicalized for result in generated_results)
                / max(len(generated_results), 1)
            ),
            device=self.accelerator.device,
        )
        lengths = gather(local_length)
        truncated = gather(local_truncated)
        canonicalized = gather(local_canonicalized)
        self._reward_metrics["completions/mean_length"].append(lengths.mean().item())
        self._reward_metrics["completions/min_length"].append(lengths.min().item())
        self._reward_metrics["completions/max_length"].append(lengths.max().item())
        self._reward_metrics["completions/clipped_ratio"].append(truncated.mean().item())
        self._reward_metrics["completions/canonicalized_ratio"].append(
            canonicalized.mean().item()
        )

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 1:
            raise ValueError("Expected one raw sample per rank for interactive rollout.")
        sample = inputs[0]
        group_sample_ids = gather_object([str(sample["id"])])
        if len(group_sample_ids) != self.num_generations or len(set(group_sample_ids)) != 1:
            raise RuntimeError(
                "GRPO group members must be eight generations of the same sample, "
                f"got sample IDs: {group_sample_ids}"
            )
        rollout = self._rollout(sample)
        reward = rollout["reward"]
        local_reward = torch.tensor([reward.total_reward], device=self.accelerator.device)
        group_rewards = gather(local_reward)
        if group_rewards.numel() != self.num_generations:
            raise RuntimeError(
                f"Expected {self.num_generations} gathered rewards, got {group_rewards.numel()}."
            )
        if not torch.isfinite(group_rewards).all():
            raise RuntimeError(f"GRPO group contains non-finite rewards: {group_rewards}")
        advantage = local_reward - group_rewards.mean()
        if self.args.scale_rewards:
            advantage = advantage / (group_rewards.std(unbiased=True) + 1e-4)
        self._record_reward_metrics(reward, group_rewards)
        predicted_letter, _ = parse_answer_letter(
            rollout["final_output"],
            len(sample["options"]),
        )
        group_letters = gather_object([predicted_letter or "INVALID"])
        self._reward_metrics["diversity/unique_answers"].append(
            float(len(set(group_letters)))
        )
        self._reward_metrics["diversity/answer_variation"].append(
            float(len(set(group_letters)) > 1)
        )
        self._record_completion_metrics(rollout["generation_results"])
        completion_text = "\n".join(
            rollout["action_outputs"] + [rollout["final_output"]]
        )
        gathered_logs = gather_object([{
            "sample_id": sample["id"],
            "prompt": f"{sample['Question']}\n" + "\n".join(sample["options"]),
            "completion": completion_text,
            "answer_accuracy": reward.answer_accuracy,
            "viewpoint": reward.mean_viewpoint,
            "bbox_transition_count": reward.bbox_transition_count,
            "bbox_fallback_transition_count": reward.bbox_fallback_transition_count,
            "total_reward": reward.total_reward,
            "advantage": advantage.item(),
        }])
        self._textual_logs.extend(gathered_logs)

        model_inputs = self._processor_inputs(
            rollout["messages"],
            add_generation_prompt=False,
            budget_video_turns=len(sample["video_items"]),
        )
        policy_labels = self._assistant_labels(
            model_inputs["input_ids"],
            rollout["generation_results"],
        )
        model_inputs["policy_labels"] = policy_labels
        model_inputs["advantages"] = advantage
        model_inputs["old_per_token_logps"] = None
        self.state.num_input_tokens_seen += gather(
            model_inputs["attention_mask"].sum().to(self.accelerator.device)
        ).sum().item()
        self._reward_metrics["num_tokens"] = [self.state.num_input_tokens_seen]
        return super()._prepare_inputs(model_inputs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("CamVLMInteractiveGRPOTrainer does not support return_outputs=True.")
        labels = inputs.pop("policy_labels")
        advantages = inputs.pop("advantages")
        old_per_token_logps = inputs.pop("old_per_token_logps")
        target_positions, logit_positions = policy_target_and_logit_positions(labels)
        outputs = model(
            **inputs,
            use_cache=False,
            logits_to_keep=logit_positions,
        )
        logits = outputs.logits.float()
        target_ids = inputs["input_ids"][:, target_positions]
        logits = logits / self.args.temperature
        token_advantages = advantages.reshape(-1, 1).expand_as(target_ids).to(logits.dtype)
        completion_mask = torch.ones_like(target_ids, dtype=logits.dtype)
        per_token_logps = selective_log_softmax(logits, target_ids)
        loss, clip_metrics = compute_bnpo_policy_loss(
            per_token_logps,
            old_per_token_logps,
            completion_mask,
            token_advantages,
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            delta=self.args.delta,
        )
        for key, value in clip_metrics.items():
            gathered = gather(value.detach())
            self._reward_metrics[key].append(gathered.nanmean().item())
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        for key, values in self._reward_metrics.items():
            if values:
                logs[key] = sum(values) / len(values)
        self._reward_metrics.clear()
        try:
            result = super().log(logs, start_time=start_time)
        except TypeError:
            result = super().log(logs)
        if self.accelerator.is_main_process and self.args.log_completions and self._textual_logs:
            report_to = self.args.report_to or []
            if isinstance(report_to, str):
                report_to = [report_to]
            if "wandb" in report_to:
                try:
                    import wandb

                    if wandb.run is not None:
                        columns = list(self._textual_logs[0])
                        data = [[row.get(column) for column in columns] for row in self._textual_logs]
                        wandb.log({"completions": wandb.Table(columns=columns, data=data)})
                except ImportError:
                    pass
        self._textual_logs.clear()
        return result

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint_dir = os.path.join(
            self.args.output_dir,
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        if self.accelerator.is_main_process:
            self.processor.save_pretrained(checkpoint_dir)
