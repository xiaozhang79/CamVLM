#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/ckpts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
MODE="${MODE:-camvlm_action_only}"
MODEL_PATH="${MODEL_PATH:-${CKPTS_DIR}/CamVLM-RL}"
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-${MODEL_PATH}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/datasets/udvideoqa}"
QUESTION_FILES=(
    "${QUESTION_FILE_SET20:-${DATA_ROOT}/Set_20/2.37pm_10.1pm_clips_60_annotations.jsonl}"
    "${QUESTION_FILE_SET03:-${DATA_ROOT}/Set_03/2.26pm_10.1mins_clips_annotations.jsonl}"
)
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
for question_file in "${QUESTION_FILES[@]}"; do
    if [[ ! -f "${question_file}" ]]; then
        echo "Missing required file: ${question_file}" >&2
        exit 1
    fi
done

MODEL_BASENAME="$(basename "${MODEL_PATH%/}")"
MODEL_TAG="${MODEL_TAG:-$(printf '%s-%s' "${MODEL_BASENAME}" "${MODE}" | sed 's/[^A-Za-z0-9._-]/_/g')}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_BASE_ROOT="${OUTPUT_BASE_ROOT:-${PROJECT_ROOT}/eval_results/udvideoqa}"
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
    case "${MODE}" in
        camvlm_action_only)
            OUTPUT_ROOT="${OUTPUT_BASE_ROOT}/dynamic"
            ;;
        *)
            OUTPUT_ROOT="${OUTPUT_BASE_ROOT}/passive"
            ;;
    esac
fi
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_TAG}/${RUN_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/predictions.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-${OUTPUT_DIR}/predictions.jsonl}"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/eval.log}"
SCORED_FILE="${SCORED_FILE:-${OUTPUT_DIR}/predictions.gpt_scored.json}"
SUMMARY_FILE="${SUMMARY_FILE:-${OUTPUT_DIR}/predictions.gpt_summary.json}"
CACHE_FILE="${CACHE_FILE:-${OUTPUT_DIR}/predictions.gpt_score_cache.json}"

export VIDEO_FPS="${VIDEO_FPS:-2}"
export VIDEO_MIN_TOKENS="${VIDEO_MIN_TOKENS:-64}"
export VIDEO_TOTAL_TOKENS="${VIDEO_TOTAL_TOKENS:-14336}"

INITIAL_WINDOW="${INITIAL_WINDOW:-0.3333,0.3333,0.6667,0.6667}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
FINAL_MAX_NEW_TOKENS="${FINAL_MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
RESUME="${RESUME:-0}"
DISABLE_BALANCED_CHUNKS="${DISABLE_BALANCED_CHUNKS:-0}"
QUIET="${QUIET:-0}"
USE_VLLM="${USE_VLLM:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-21504}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_VIDEO_CLIPS="${VLLM_MAX_VIDEO_CLIPS:-64}"
VLLM_TOP_P="${VLLM_TOP_P:-1.0}"
SCORE_WITH_GPT="${SCORE_WITH_GPT:-1}"
GPT_GRADER_MODEL="${GPT_GRADER_MODEL:-gpt-5.5}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_API_KEY_ENV="${OPENAI_API_KEY_ENV:-OPENAI_API_KEY}"
GPT_SCORE_WORKERS="${GPT_SCORE_WORKERS:-1024}"
GPT_SCORE_MAX_INFLIGHT="${GPT_SCORE_MAX_INFLIGHT:-128}"
GPT_SCORE_REQUESTS_PER_MINUTE="${GPT_SCORE_REQUESTS_PER_MINUTE:-0}"
GPT_SCORE_MIN_REQUESTS_PER_MINUTE="${GPT_SCORE_MIN_REQUESTS_PER_MINUTE:-10}"
GPT_SCORE_RATE_LIMIT_COOLDOWN="${GPT_SCORE_RATE_LIMIT_COOLDOWN:-15}"
GPT_SCORE_RATE_LIMIT_BACKOFF="${GPT_SCORE_RATE_LIMIT_BACKOFF:-0.7}"
GPT_SCORE_MAX_OUTPUT_TOKENS="${GPT_SCORE_MAX_OUTPUT_TOKENS:-32}"
GPT_REASONING_EFFORT="${GPT_REASONING_EFFORT:-none}"
REDIRECT_CHUNK_LOGS="${REDIRECT_CHUNK_LOGS:-0}"

if [[ "${USE_VLLM}" == "1" ]]; then
    export VLLM_USE_V1="${VLLM_USE_V1:-1}"
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
AVAILABLE_GPUS="${#GPU_LIST[@]}"
if [[ "${AVAILABLE_GPUS}" -le 0 ]]; then
    echo "CUDA_VISIBLE_DEVICES must contain at least one GPU id." >&2
    exit 1
fi

ACTIVE_GPUS="${EVAL_NUM_GPUS:-${AVAILABLE_GPUS}}"
if ! [[ "${ACTIVE_GPUS}" =~ ^[0-9]+$ ]] || [[ "${ACTIVE_GPUS}" -le 0 ]]; then
    echo "EVAL_NUM_GPUS must be a positive integer, got: ${ACTIVE_GPUS}" >&2
    exit 1
fi
if [[ "${ACTIVE_GPUS}" -gt "${AVAILABLE_GPUS}" ]]; then
    echo "EVAL_NUM_GPUS=${ACTIVE_GPUS} exceeds available CUDA_VISIBLE_DEVICES count ${AVAILABLE_GPUS}." >&2
    exit 1
fi
GPU_LIST=("${GPU_LIST[@]:0:${ACTIVE_GPUS}}")
NUM_GPUS="${#GPU_LIST[@]}"
NUM_CHUNKS="${NUM_CHUNKS:-${NUM_GPUS}}"
if [[ "${NUM_CHUNKS}" -ne "${NUM_GPUS}" ]]; then
    echo "NUM_CHUNKS must equal GPU count (${NUM_GPUS}) for this runner." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/chunks"
FINAL_LOG_FILE="${LOG_FILE}"
LOG_FILE="/dev/null"
: > "${FINAL_LOG_FILE}"

write_final_log() {
    if [[ ! -f "${SUMMARY_FILE}" ]]; then
        : > "${FINAL_LOG_FILE}"
        return
    fi
    env "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c '
import json
import sys
from evaluation.udvideoqa.score import format_log_summary

with open(sys.argv[1], encoding="utf-8") as handle:
    print(format_log_summary(json.load(handle)))
' "${SUMMARY_FILE}" > "${FINAL_LOG_FILE}"
}
echo "MODE=${MODE}" | tee -a "${LOG_FILE}"
echo "MODEL_PATH=${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "VLLM_MODEL_PATH=${VLLM_MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "DATA_ROOT=${DATA_ROOT}" | tee -a "${LOG_FILE}"
printf 'QUESTION_FILES=%s\n' "${QUESTION_FILES[*]}" | tee -a "${LOG_FILE}"
echo "MODEL_TAG=${MODEL_TAG}" | tee -a "${LOG_FILE}"
echo "RUN_ID=${RUN_ID}" | tee -a "${LOG_FILE}"
echo "OUTPUT_DIR=${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
echo "OUTPUT_FILE=${OUTPUT_FILE}" | tee -a "${LOG_FILE}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}" | tee -a "${LOG_FILE}"
echo "PYTHON_BIN=${PYTHON_BIN}" | tee -a "${LOG_FILE}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${LOG_FILE}"
echo "AVAILABLE_GPUS=${AVAILABLE_GPUS}" | tee -a "${LOG_FILE}"
echo "ACTIVE_GPUS=${NUM_GPUS}" | tee -a "${LOG_FILE}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${LOG_FILE}"
echo "NUM_CHUNKS=${NUM_CHUNKS}" | tee -a "${LOG_FILE}"
echo "HF_HOME=${HF_HOME:-}" | tee -a "${LOG_FILE}"
echo "VIDEO_FPS=${VIDEO_FPS}" | tee -a "${LOG_FILE}"
echo "VIDEO_MIN_TOKENS=${VIDEO_MIN_TOKENS}" | tee -a "${LOG_FILE}"
echo "VIDEO_TOTAL_TOKENS=${VIDEO_TOTAL_TOKENS}" | tee -a "${LOG_FILE}"
echo "INITIAL_WINDOW=${INITIAL_WINDOW}" | tee -a "${LOG_FILE}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS}" | tee -a "${LOG_FILE}"
echo "FINAL_MAX_NEW_TOKENS=${FINAL_MAX_NEW_TOKENS}" | tee -a "${LOG_FILE}"
echo "USE_VLLM=${USE_VLLM}" | tee -a "${LOG_FILE}"
echo "SCORE_WITH_GPT=${SCORE_WITH_GPT}" | tee -a "${LOG_FILE}"
echo "GPT_GRADER_MODEL=${GPT_GRADER_MODEL}" | tee -a "${LOG_FILE}"
echo "OPENAI_BASE_URL=${OPENAI_BASE_URL}" | tee -a "${LOG_FILE}"
echo "OPENAI_API_KEY_ENV=${OPENAI_API_KEY_ENV}" | tee -a "${LOG_FILE}"
echo "GPT_SCORE_WORKERS=${GPT_SCORE_WORKERS}" | tee -a "${LOG_FILE}"
echo "GPT_REASONING_EFFORT=${GPT_REASONING_EFFORT}" | tee -a "${LOG_FILE}"
if [[ "${USE_VLLM}" == "1" ]]; then
    echo "VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION}" | tee -a "${LOG_FILE}"
    echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}" | tee -a "${LOG_FILE}"
    echo "VLLM_MAX_VIDEO_CLIPS=${VLLM_MAX_VIDEO_CLIPS}" | tee -a "${LOG_FILE}"
fi

cd "${PROJECT_ROOT}"
pids=()
chunk_files=()
chunk_logs=()
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id="${GPU_LIST[$gpu_idx]}"
    chunk_idx="${gpu_idx}"
    chunk_file="${OUTPUT_DIR}/chunks/${NUM_CHUNKS}_${chunk_idx}.jsonl"
    chunk_log="${OUTPUT_DIR}/chunks/${NUM_CHUNKS}_${chunk_idx}.log"
    chunk_files+=("${chunk_file}")
    chunk_logs+=("${chunk_log}")

    resume_args=()
    if [[ "${RESUME}" == "1" ]]; then
        resume_args+=(--resume)
    fi
    chunk_args=()
    if [[ "${DISABLE_BALANCED_CHUNKS}" == "1" ]]; then
        chunk_args+=(--disable_balanced_chunks)
    fi
    quiet_args=()
    if [[ "${QUIET}" == "1" ]]; then
        quiet_args+=(--quiet)
    fi
    vllm_args=()
    if [[ "${USE_VLLM}" == "1" ]]; then
        vllm_args+=(
            --use_vllm
            --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
            --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}"
            --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}"
            --vllm_max_video_clips "${VLLM_MAX_VIDEO_CLIPS}"
            --vllm_top_p "${VLLM_TOP_P}"
        )
    fi

    echo "Launching GPU ${gpu_id} for chunk ${chunk_idx}/${NUM_CHUNKS}"
    command=(
        env
        "CUDA_VISIBLE_DEVICES=${gpu_id}"
        "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}"
        "${PYTHON_BIN}" -m evaluation.udvideoqa.infer
        --mode "${MODE}"
        --model_path "${MODEL_PATH}"
        --vllm_model_path "${VLLM_MODEL_PATH}"
        --question_files "${QUESTION_FILES[@]}"
        --output_file "${chunk_file}"
        --initial_window "${INITIAL_WINDOW}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --final_max_new_tokens "${FINAL_MAX_NEW_TOKENS}"
        --total_video_tokens "${VIDEO_TOTAL_TOKENS}"
        --temperature "${TEMPERATURE}"
        --torch_dtype "${TORCH_DTYPE}"
        "${vllm_args[@]}"
        --num_chunks "${NUM_CHUNKS}"
        --chunk_idx "${chunk_idx}"
        --progress_position "${chunk_idx}"
        --max_samples "${MAX_SAMPLES}"
        "${chunk_args[@]}"
        "${quiet_args[@]}"
        "${resume_args[@]}"
    )
    if [[ "${REDIRECT_CHUNK_LOGS}" == "1" ]]; then
        "${command[@]}" > "${chunk_log}" 2>&1 &
    else
        "${command[@]}" &
    fi
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if [[ "${REDIRECT_CHUNK_LOGS}" == "1" ]]; then
    rm -f "${chunk_logs[@]}"
fi
if [[ "${failed}" != "0" ]]; then
    echo "At least one UDVideoQA inference chunk failed." >&2
    exit 1
fi

merge_command=(
    env
    "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m evaluation.udvideoqa.merge_results
    --inputs "${chunk_files[@]}"
    --output "${OUTPUT_FILE}"
    --jsonl-output "${OUTPUT_JSONL}"
    --question_files "${QUESTION_FILES[@]}"
)
"${merge_command[@]}" 2>&1 | tee -a "${LOG_FILE}"

if [[ "${SCORE_WITH_GPT}" == "1" ]]; then
    if [[ -z "${!OPENAI_API_KEY_ENV:-}" ]]; then
        echo "${OPENAI_API_KEY_ENV} is not set; skipping GPT scoring. Set SCORE_WITH_GPT=0 to silence this." | tee -a "${LOG_FILE}"
    else
        score_command=(
            env
            "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}"
            "${PYTHON_BIN}" -m evaluation.udvideoqa.score
            --input "${OUTPUT_FILE}"
            --output "${SCORED_FILE}"
            --summary-output "${SUMMARY_FILE}"
            --cache "${CACHE_FILE}"
            --model "${GPT_GRADER_MODEL}"
            --base-url "${OPENAI_BASE_URL}"
            --api-key-env "${OPENAI_API_KEY_ENV}"
            --workers "${GPT_SCORE_WORKERS}"
            --max-inflight "${GPT_SCORE_MAX_INFLIGHT}"
            --requests-per-minute "${GPT_SCORE_REQUESTS_PER_MINUTE}"
            --min-requests-per-minute "${GPT_SCORE_MIN_REQUESTS_PER_MINUTE}"
            --rate-limit-cooldown "${GPT_SCORE_RATE_LIMIT_COOLDOWN}"
            --rate-limit-backoff "${GPT_SCORE_RATE_LIMIT_BACKOFF}"
            --max-output-tokens "${GPT_SCORE_MAX_OUTPUT_TOKENS}"
            --reasoning-effort "${GPT_REASONING_EFFORT}"
        )
        "${score_command[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
fi

write_final_log

echo "Done. Predictions: ${OUTPUT_FILE}"
echo "JSONL Predictions: ${OUTPUT_JSONL}"
echo "Log: ${FINAL_LOG_FILE}"
