#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/ckpts}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-${CKPTS_DIR}/CamVLM-SFT}"
DATASET_NAME="${DATASET_NAME:-${PROJECT_ROOT}/datasets/camtrack_53k/train_rl.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPTS_DIR}/CamVLM-RL}"
RUN_NAME="${RUN_NAME:-CamVLM-RL}"
MASTER_PORT="${MASTER_PORT:-12411}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.36}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-24576}"

export VLLM_USE_V1=1
export WANDB_PROJECT="${WANDB_PROJECT:-CamVLM-RL}"
export CAMTRACK_JPEG_READ_WORKERS="${CAMTRACK_JPEG_READ_WORKERS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Missing SFT model config: ${MODEL_PATH}/config.json" >&2
    echo "Set MODEL_PATH to a completed 1/9 SFT checkpoint." >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
torchrun --nproc_per_node=8 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    -m training.rl.train_rl \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --output_dir "${OUTPUT_DIR}" \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2.json" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --tf32 True \
    --gradient_checkpointing True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --num_train_epochs 1 \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 25 \
    --save_total_limit 100 \
    --save_only_model True \
    --dataloader_num_workers 8 \
    --log_completions True \
    --num_generations 8 \
    --steps_per_generation 1 \
    --temperature 1.0 \
    --max_action_new_tokens 96 \
    --max_final_new_tokens 64 \
    --max_completion_length 768 \
    --scale_rewards False \
    --reward_funcs accuracy,viewpoint \
    --fps 2 \
    --min_tokens 64 \
    --total_tokens 14336 \
    --max_frames 448 \
    --mevis_fps 6 \
    --youtube_vos_fps 6 \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_max_video_clips 64 \
    --seed 42 \
    --report_to wandb \
    --logging_dir wandb \
    --run_name "${RUN_NAME}" \
    --remove_unused_columns False
