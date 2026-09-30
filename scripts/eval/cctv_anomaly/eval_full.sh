#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPTS_DIR="${CKPTS_DIR:-${PROJECT_ROOT}/ckpts}"
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
DATA="${DATA:-${PROJECT_ROOT}/datasets/cctv_anomaly/test.jsonl}"
MODEL_PATH="${MODEL_PATH:-${CKPTS_DIR}/CamVLM-RL}"
if [[ ! -f "${DATA}" ]]; then
    echo "Missing required file: ${DATA}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
    echo "Missing model config: ${MODEL_PATH}/config.json" >&2
    exit 1
fi
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_PATH%/}")-full}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/eval_results/cctv_anomaly/full}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${MODEL_TAG}/${RUN_ID}}"
OUTPUT_FILE="${OUTPUT_DIR}/predictions.json"
OUTPUT_JSONL="${OUTPUT_DIR}/predictions.jsonl"
SUMMARY_FILE="${OUTPUT_DIR}/predictions.gpt_summary.json"
LOG_FILE="${OUTPUT_DIR}/eval.log"
VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MIN_TOKENS="${VIDEO_MIN_TOKENS:-64}"
VIDEO_TOTAL_TOKENS="${VIDEO_TOTAL_TOKENS:-14336}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_FRAMES="${MAX_FRAMES:-448}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
RESUME="${RESUME:-0}"
VLLM_TOP_P="${VLLM_TOP_P:-1.0}"
USE_VLLM="${USE_VLLM:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-21504}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
VLLM_MAX_VIDEO_CLIPS="${VLLM_MAX_VIDEO_CLIPS:-64}"
SCORE_WITH_GPT="${SCORE_WITH_GPT:-1}"
GPT_GRADER_MODEL="${GPT_GRADER_MODEL:-gpt-5.5}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
GPT_SCORE_WORKERS="${GPT_SCORE_WORKERS:-512}"
GPT_SCORE_MAX_INFLIGHT="${GPT_SCORE_MAX_INFLIGHT:-128}"
GPT_SCORE_REQUESTS_PER_MINUTE="${GPT_SCORE_REQUESTS_PER_MINUTE:-0}"

export VIDEO_FPS VIDEO_MIN_TOKENS VIDEO_TOTAL_TOKENS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-ERROR}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"; fi
IFS=',' read -ra GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${EVAL_NUM_GPUS:-${#GPU_LIST[@]}}"
if (( NUM_GPUS > ${#GPU_LIST[@]} )); then NUM_GPUS=${#GPU_LIST[@]}; fi
mkdir -p "${OUTPUT_DIR}/chunks"
: > "${LOG_FILE}"

pids=()
chunk_files=()
for idx in $(seq 0 $((NUM_GPUS - 1))); do
    chunk="${OUTPUT_DIR}/chunks/${NUM_GPUS}_${idx}.jsonl"
    chunk_files+=("${chunk}")
    command=(env "CUDA_VISIBLE_DEVICES=${GPU_LIST[$idx]}" "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m evaluation.cctv_anomaly.infer --model-path "${MODEL_PATH}" --data "${DATA}" --output-file "${chunk}" --video-fps "${VIDEO_FPS}" --min-video-tokens "${VIDEO_MIN_TOKENS}" --max-frames "${MAX_FRAMES}" --total-video-tokens "${VIDEO_TOTAL_TOKENS}" --max-new-tokens "${MAX_NEW_TOKENS}" --temperature "${TEMPERATURE}" --vllm-top-p "${VLLM_TOP_P}" --num-chunks "${NUM_GPUS}" --chunk-idx "${idx}" --max-samples "${MAX_SAMPLES}" --progress-position "${idx}")
    if [[ "${USE_VLLM}" == "1" ]]; then command+=(--use-vllm --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}" --vllm-max-num-seqs "${VLLM_MAX_NUM_SEQS}" --vllm-max-video-clips "${VLLM_MAX_VIDEO_CLIPS}"); fi
    if [[ "${RESUME}" == "1" ]]; then command+=(--resume); fi
    "${command[@]}" & pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
merge_command=(
    env "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}"
    -m evaluation.cctv_anomaly.merge_results
    --inputs "${chunk_files[@]}"
    --data "${DATA}"
    --output "${OUTPUT_FILE}"
    --jsonl-output "${OUTPUT_JSONL}"
)
"${merge_command[@]}"
if [[ "${SCORE_WITH_GPT}" == "1" ]]; then
    env "PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -m evaluation.cctv_anomaly.score --input "${OUTPUT_FILE}" --summary-output "${SUMMARY_FILE}" --model "${GPT_GRADER_MODEL}" --base-url "${OPENAI_BASE_URL}" --workers "${GPT_SCORE_WORKERS}" --max-inflight "${GPT_SCORE_MAX_INFLIGHT}" --requests-per-minute "${GPT_SCORE_REQUESTS_PER_MINUTE}" --max-output-tokens "${GPT_SCORE_MAX_OUTPUT_TOKENS:-1600}" | tail -n 1 > "${LOG_FILE}"
fi
