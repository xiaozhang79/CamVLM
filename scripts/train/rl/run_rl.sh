#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/ckpts}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-${CKPTS_DIR}/CamVLM-SFT}"
DATASET_NAME="${DATASET_NAME:-${PROJECT_ROOT}/datasets/camtrack_53k/train_rl.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPTS_DIR}/CamVLM-RL}"
RUN_NAME="${RUN_NAME:-CamVLM-RL}"
MASTER_PORT="${MASTER_PORT:-12401}"

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
    --deepspeed "${PROJECT_ROOT}/scripts/zero1.json" \
    --bf16 True \
    --tf32 True \
    --gradient_checkpointing True \
    --freeze_vision_tower True \
    --freeze_llm False \
    --freeze_merger False \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-6 \
    --lr_scheduler_type constant \
    --max_steps 2000 \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 10 \
    --save_only_model True \
    --dataloader_num_workers 8 \
    --log_completions True \
    --max_action_new_tokens 96 \
    --max_final_new_tokens 64 \
    --max_completion_length 768 \
    --reward_funcs accuracy,viewpoint \
    --min_tokens 64 \
    --total_tokens 14336 \
    --max_frames 448 \
    --seed 42 \
    --report_to wandb \
    --logging_dir wandb \
    --run_name "${RUN_NAME}"
