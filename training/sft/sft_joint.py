import copy
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Sampler
from trl import ModelConfig, SFTConfig, SFTTrainer, ScriptArguments, TrlParser
from transformers import AutoProcessor


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.sft import sft_cctv_anomaly as cctv_sft
from training.sft import sft_camtrack as camtrack_sft


@dataclass
class JointSFTArguments(ScriptArguments):
    cctv_dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Local CCTV-Anomaly JSONL used for full-video category and description SFT."}
    )
    camtrack_per_cctv: int = field(
        default=3,
        metadata={"help": "Number of CamTrack samples per CCTV sample in each global batch."},
    )
    camtrack_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional cap for valid CamTrack samples. When set, samples are selected "
            "stratified by source with camtrack_subset_seed."
        },
    )
    camtrack_subset_seed: int = field(
        default=42,
        metadata={"help": "Random seed for deterministic stratified CamTrack sampling."},
    )


class JointDataset(TorchDataset):
    def __init__(self, camtrack_dataset: Dataset, cctv_dataset: Dataset):
        self.camtrack_dataset = camtrack_dataset
        self.cctv_dataset = cctv_dataset
        self.camtrack_size = len(camtrack_dataset)
        self.cctv_size = len(cctv_dataset)

    def __len__(self) -> int:
        return self.camtrack_size + self.cctv_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < self.camtrack_size:
            sample = dict(self.camtrack_dataset[index])
            sample["_joint_task"] = "camtrack"
            return sample
        sample = dict(self.cctv_dataset[index - self.camtrack_size])
        sample["_joint_task"] = "cctv"
        return sample


class TaskRatioSampler(Sampler[int]):
    """Build global microbatches with a fixed CamTrack:CCTV sample ratio.

    Accelerate shards the resulting stream across ranks. With the default
    1:1 setting on eight GPUs, every global microbatch contains four CamTrack
    and four CCTV samples.
    """

    def __init__(
        self,
        dataset: JointDataset,
        world_size: int,
        camtrack_per_cctv: int,
        seed: int,
    ) -> None:
        if dataset.camtrack_size == 0 or dataset.cctv_size == 0:
            raise ValueError("Joint SFT requires non-empty CamTrack and CCTV datasets.")
        if camtrack_per_cctv < 1:
            raise ValueError("camtrack_per_cctv must be at least 1.")
        self.dataset = dataset
        self.world_size = max(1, int(world_size))
        self.camtrack_per_cctv = int(camtrack_per_cctv)
        self.seed = int(seed)
        self.epoch = 0

        if self.world_size == 1:
            self.camtrack_per_global_batch = self.camtrack_per_cctv
            self.cctv_per_global_batch = 1
        else:
            ratio_total = self.camtrack_per_cctv + 1
            self.cctv_per_global_batch = max(1, round(self.world_size / ratio_total))
            self.cctv_per_global_batch = min(self.cctv_per_global_batch, self.world_size - 1)
            self.camtrack_per_global_batch = self.world_size - self.cctv_per_global_batch

        self.global_batch_size = self.camtrack_per_global_batch + self.cctv_per_global_batch
        self.num_global_batches = math.ceil(
            self.dataset.camtrack_size / self.camtrack_per_global_batch
        )

    def __len__(self) -> int:
        return self.num_global_batches * self.global_batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        rng = random.Random(self.seed + self.epoch)
        camtrack_indices = list(range(self.dataset.camtrack_size))
        cctv_indices = list(range(self.dataset.cctv_size))
        rng.shuffle(camtrack_indices)
        rng.shuffle(cctv_indices)

        for batch_index in range(self.num_global_batches):
            task_slots = (
                ["camtrack"] * self.camtrack_per_global_batch
                + ["cctv"] * self.cctv_per_global_batch
            )
            rng.shuffle(task_slots)
            camtrack_offset = batch_index * self.camtrack_per_global_batch
            cctv_offset = batch_index * self.cctv_per_global_batch
            for task in task_slots:
                if task == "camtrack":
                    yield camtrack_indices[camtrack_offset % self.dataset.camtrack_size]
                    camtrack_offset += 1
                else:
                    cctv_index = cctv_indices[cctv_offset % self.dataset.cctv_size]
                    yield self.dataset.camtrack_size + cctv_index
                    cctv_offset += 1


class JointSFTTrainer(SFTTrainer):
    def __init__(self, *args, camtrack_per_cctv: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.camtrack_per_cctv = camtrack_per_cctv

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if not isinstance(dataset, JointDataset):
            raise TypeError("JointSFTTrainer requires JointDataset.")
        return TaskRatioSampler(
            dataset,
            world_size=self.args.world_size,
            camtrack_per_cctv=self.camtrack_per_cctv,
            seed=self.args.seed,
        )


processor = None


def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    if len(examples) != 1:
        raise ValueError(
            "Joint SFT requires per_device_train_batch_size=1 so each task keeps its original collator."
        )
    example = copy.deepcopy(examples[0])
    task = example.pop("_joint_task", None)
    if task == "cctv":
        return cctv_sft.collate_fn([example])
    if task == "camtrack":
        previous_template = processor.chat_template
        try:
            processor.chat_template = camtrack_sft.QWEN3_CAMTRACK_CHAT_TEMPLATE
            return camtrack_sft.collate_fn([example])
        finally:
            processor.chat_template = previous_template
    raise ValueError(f"Unknown Joint SFT task: {task!r}")


def _camtrack_source(row: Dict[str, Any]) -> str:
    path = str(row.get("path", ""))
    if "A2D-Sentences" in path:
        return "A2D"
    if "MeViS" in path:
        return "MeViS"
    if "Refer-YouTube-VOS" in path:
        return "YouTube-VOS"
    return "other"


def _stratified_camtrack_subset(
    rows: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int,
) -> List[Dict[str, Any]]:
    if max_samples is None:
        return rows
    if max_samples < 1:
        raise ValueError("camtrack_max_samples must be positive when provided.")
    if max_samples > len(rows):
        raise ValueError(
            f"camtrack_max_samples={max_samples} exceeds valid CamTrack samples={len(rows)}."
        )
    if max_samples == len(rows):
        return rows

    rows_by_source: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_source.setdefault(_camtrack_source(row), []).append(row)

    total = len(rows)
    quotas = {
        source: int(len(source_rows) * max_samples // total)
        for source, source_rows in rows_by_source.items()
    }
    remaining = max_samples - sum(quotas.values())
    source_order = sorted(
        rows_by_source,
        key=lambda source: (
            -(len(rows_by_source[source]) * max_samples % total),
            source,
        ),
    )
    for source in source_order[:remaining]:
        quotas[source] += 1

    selected: List[Dict[str, Any]] = []
    for source in sorted(rows_by_source):
        source_rows = list(rows_by_source[source])
        random.Random(f"{seed}:{source}").shuffle(source_rows)
        selected.extend(source_rows[:quotas[source]])
    random.Random(seed).shuffle(selected)
    if len(selected) != max_samples:
        raise RuntimeError(
            f"Stratified CamTrack selection returned {len(selected)} samples, expected {max_samples}."
        )
    return selected


def _load_camtrack_dataset(
    path: str,
    max_samples: Optional[int] = None,
    subset_seed: int = 42,
) -> Dataset:
    if not path.endswith(".json"):
        raise ValueError("Joint SFT CamTrack data must be a local JSON file.")
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    original_count = len(rows)
    rows = [row for row in rows if camtrack_sft._has_complete_second_video_turn(row)]
    if not rows:
        raise ValueError(f"No valid CamTrack samples in {path}.")
    dropped_count = original_count - len(rows)
    if dropped_count:
        print(
            f"Dropped CamTrack samples without a complete 1-second video turn: {dropped_count}/{original_count}",
            flush=True,
        )
    rows = _stratified_camtrack_subset(rows, max_samples=max_samples, seed=subset_seed)
    if max_samples is not None:
        selected_by_source: Dict[str, int] = {}
        for row in rows:
            source = _camtrack_source(row)
            selected_by_source[source] = selected_by_source.get(source, 0) + 1
        print(
            f"Selected {len(rows)} valid CamTrack samples with stratified source mix: "
            f"{dict(sorted(selected_by_source.items()))}",
            flush=True,
        )
    stats = camtrack_sft._mark_action_supervision(rows)
    print(f"CamTrack action supervision: {stats}", flush=True)
    return Dataset.from_list(rows)


def _load_cctv_dataset(path: str) -> Dataset:
    if not path.endswith(".jsonl"):
        raise ValueError("Joint SFT CCTV data must be a local JSONL file.")
    dataset = Dataset.from_json(path)
    cctv_sft._validate_dataset(dataset)
    return dataset


def _configure_task_collators(shared_processor) -> None:
    global processor
    processor = shared_processor
    cctv_sft.processor = shared_processor
    camtrack_sft.processor = shared_processor


def _cleanup_distributed() -> None:
    camtrack_sft._distributed_barrier_with_device()
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = TrlParser((JointSFTArguments, SFTConfig, ModelConfig, camtrack_sft.SFTFreezeConfig))
    script_args, training_args, model_config, freeze_config = parser.parse_args_and_config()
    if training_args.per_device_train_batch_size != 1:
        raise ValueError("Joint SFT currently requires --per_device_train_batch_size 1.")

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}
    training_args.freeze_llm = freeze_config.freeze_llm
    training_args.freeze_vision_tower = freeze_config.freeze_vision_tower
    training_args.freeze_merger = freeze_config.freeze_merger
    if not script_args.dataset_name:
        raise ValueError("Joint SFT requires --dataset_name for the CamTrack dataset.")
    if not script_args.cctv_dataset_name:
        raise ValueError("Joint SFT requires --cctv_dataset_name for the CCTV dataset.")

    camtrack_dataset = _load_camtrack_dataset(
        script_args.dataset_name,
        max_samples=script_args.camtrack_max_samples,
        subset_seed=script_args.camtrack_subset_seed,
    )
    cctv_dataset = _load_cctv_dataset(script_args.cctv_dataset_name)
    dataset = JointDataset(camtrack_dataset, cctv_dataset)
    world_size = max(1, int(training_args.world_size))
    cctv_per_global_batch = max(1, round(world_size / (script_args.camtrack_per_cctv + 1)))
    camtrack_per_global_batch = world_size - cctv_per_global_batch if world_size > 1 else script_args.camtrack_per_cctv
    print(
        "Joint SFT datasets: "
        f"CamTrack={len(camtrack_dataset)}, CCTV={len(cctv_dataset)}, "
        f"global_batch_mix={camtrack_per_global_batch}:{cctv_per_global_batch}",
        flush=True,
    )

    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
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
    model = camtrack_sft._load_model(model_config.model_name_or_path, model_kwargs)
    model.config.use_cache = False
    camtrack_sft.configure_llm(model, training_args)
    camtrack_sft.configure_vision_tower(
        model,
        training_args,
        camtrack_sft._resolve_compute_dtype(training_args),
        training_args.device,
    )
    if training_args.local_rank in (-1, 0):
        camtrack_sft._print_trainable_parameters(model, freeze_config)

    shared_processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    _configure_task_collators(shared_processor)
    trainer = JointSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        processing_class=shared_processor,
        camtrack_per_cctv=script_args.camtrack_per_cctv,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    shared_processor.save_pretrained(training_args.output_dir)
    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)
    del model
    del trainer
    torch.cuda.empty_cache()
    _cleanup_distributed()
