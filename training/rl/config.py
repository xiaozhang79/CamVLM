from dataclasses import dataclass, field
from typing import Optional

from trl import GRPOConfig


@dataclass
class ModelArguments:
    model_name_or_path: str = field()
    trust_remote_code: bool = True
    attn_implementation: str = "flash_attention_2"


@dataclass
class DataArguments:
    dataset_name: str = field()
    fps: float = 2.0
    min_tokens: int = 64
    total_tokens: int = 14336
    max_frames: int = 448
    mevis_fps: float = 6.0
    youtube_vos_fps: float = 6.0


@dataclass
class CamVLMGRPOArguments(GRPOConfig):
    optim: str = "adamw_torch"
    disable_flash_attn2: bool = False
    double_quant: bool = True
    quant_type: str = "nf4"
    bits: int = 16
    steps_per_generation: int = 1
    scale_rewards: bool = False
    loss_type: str = "bnpo"
    beta: float = 0.0
    num_iterations: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: int = 512
    num_generations: int = 8
    max_action_new_tokens: int = 96
    max_final_new_tokens: int = 64
    initial_window: str = "0.3333,0.3333,0.6667,0.6667"
    reward_funcs: str = "accuracy,viewpoint"
    freeze_vision_tower: bool = True
    freeze_llm: bool = False
    freeze_merger: bool = False
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    vllm_max_model_len: int = 32768
    vllm_max_video_clips: int = 64
    vllm_required_model_arch: str = "Qwen3VLForConditionalGeneration"
