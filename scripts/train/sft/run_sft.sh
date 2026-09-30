#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/ckpts}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_PATH="${MODEL_PATH:-${CKPTS_DIR}/Qwen3-VL-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-${PROJECT_ROOT}/datasets/camtrack_53k/train_sft.json}"
CCTV_DATASET_NAME="${CCTV_DATASET_NAME:-${PROJECT_ROOT}/datasets/cctv_anomaly/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${CKPTS_DIR}/CamVLM-SFT}"
RUN_NAME="${RUN_NAME:-CamVLM-SFT}"

# Match CamTrack samples to the CCTV training count for a 1:1 task mixture.
CAMTRACK_PER_CCTV="${CAMTRACK_PER_CCTV:-1}"
CAMTRACK_MAX_SAMPLES="${CAMTRACK_MAX_SAMPLES:-13138}"
CAMTRACK_SUBSET_SEED="${CAMTRACK_SUBSET_SEED:-42}"

MASTER_PORT="${MASTER_PORT:-12351}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
cd "${PROJECT_ROOT}"

export CAMTRACK_SFT_FPS="${CAMTRACK_SFT_FPS:-2}"
export CAMTRACK_MEVIS_FPS="${CAMTRACK_MEVIS_FPS:-6}"
export CAMTRACK_YOUTUBE_VOS_FPS="${CAMTRACK_YOUTUBE_VOS_FPS:-6}"
export CAMTRACK_TOTAL_TOKENS="${CAMTRACK_TOTAL_TOKENS:-14336}"
export CAMTRACK_MIN_TOKENS="${CAMTRACK_MIN_TOKENS:-64}"
export CAMTRACK_MAX_FRAMES="${CAMTRACK_MAX_FRAMES:-448}"
export CAMTRACK_JPEG_READ_WORKERS="${CAMTRACK_JPEG_READ_WORKERS:-8}"
export STAGE1_MAX_FRAMES="${STAGE1_MAX_FRAMES:-448}"
export STAGE1_MIN_TOKENS="${STAGE1_MIN_TOKENS:-64}"

CONDA_ENV_LIB="$("${PYTHON:-python}" - <<'PY'
import sys
from pathlib import Path
print(Path(sys.executable).resolve().parents[1] / "lib")
PY
)"
FILTERED_LD_LIBRARY_PATH="$("${PYTHON:-python}" - <<'PY'
import os
blocked = {"/lib/x86_64-linux-gnu", "/usr/lib/x86_64-linux-gnu"}
print(":".join(path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path and path not in blocked))
PY
)"
if [[ -n "${FILTERED_LD_LIBRARY_PATH}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_ENV_LIB}:${FILTERED_LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${CONDA_ENV_LIB}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-True}"

DATALOADER_ARGS=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
if [[ "${DATALOADER_NUM_WORKERS}" -gt 0 ]]; then
    DATALOADER_ARGS+=(
        --dataloader_prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
        --dataloader_persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
    )
fi
CAMTRACK_SUBSET_ARGS=()
if [[ -n "${CAMTRACK_MAX_SAMPLES}" ]]; then
    CAMTRACK_SUBSET_ARGS+=(--camtrack_max_samples "${CAMTRACK_MAX_SAMPLES}")
fi

torchrun --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    training/sft/sft_joint.py \
    --output_dir "${OUTPUT_DIR}" \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset_name "${DATASET_NAME}" \
    --cctv_dataset_name "${CCTV_DATASET_NAME}" \
    --camtrack_per_cctv "${CAMTRACK_PER_CCTV}" \
    "${CAMTRACK_SUBSET_ARGS[@]}" \
    --camtrack_subset_seed "${CAMTRACK_SUBSET_SEED}" \
    --deepspeed "${PROJECT_ROOT}/scripts/zero2.json" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    "${DATALOADER_ARGS[@]}" \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --torch_dtype bfloat16 \
    --trust_remote_code True \
    --report_to none \
    --gradient_checkpointing True \
    --freeze_vision_tower True \
    --freeze_merger False \
    --freeze_llm False \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 1 \
    --run_name "${RUN_NAME}" \
    --save_steps 50 \
    --save_only_model True \
    --save_strategy steps \
    --save_total_limit 99 \
    --seed 42
