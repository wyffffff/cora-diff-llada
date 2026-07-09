#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cora

if [ -n "${HF_ENDPOINT:-}" ]; then
  export HF_ENDPOINT
fi
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_OUT="output/ablations/all_tasks_dapd_paper_full"
RUN_DIRECT="${RUN_DIRECT:-false}"
BLOCK_MODE="${BLOCK_MODE:-single}"

mkdir -p "$BASE_OUT"

get_block_length () {
  GEN="$1"
  if [ "$BLOCK_MODE" = "single" ]; then
    echo "$GEN"
  else
    echo 32
  fi
}

make_dapd_args () {
  GEN="$1"
  STEPS="$2"
  ALG="$3"
  TAU_MIN="$4"
  TAU_MAX="$5"
  BLOCK_LENGTH="$(get_block_length "$GEN")"
  echo "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=${GEN},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=low_confidence,cfg=0.0,method=DAPD,dapd_alg=${ALG},dapd_layer_ratio=0.3,dapd_tau_min=${TAU_MIN},dapd_tau_max=${TAU_MAX}"
}

summarize_one () {
  LOG_FILE="$1"
  echo ""
  echo "========== SUMMARY: ${LOG_FILE} =========="
  grep -E "Number of tokens|Generation time|Tokens per second|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$LOG_FILE" || true
  echo "========== END SUMMARY =========="
}

run_eval () {
  SETTING="$1"
  TASK="$2"
  FEWSHOT="$3"
  GEN="$4"
  STEPS="$5"
  ALG="$6"
  TAU_MIN="$7"
  TAU_MAX="$8"
  OUT_DIR="$9"

  ALG_TAG="${ALG#dapd_}"
  TAU_TAG="tau${TAU_MIN/./}_${TAU_MAX/./}"
  NAME="dapd_${ALG_TAG}_${TAU_TAG}_full"
  LOG_FILE="${OUT_DIR}/${NAME}.log"

  mkdir -p "$OUT_DIR"

  if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "========== SKIP ${SETTING}/${NAME}: log already exists =========="
    summarize_one "$LOG_FILE"
    return 0
  fi

  MODEL_ARGS="$(make_dapd_args "$GEN" "$STEPS" "$ALG" "$TAU_MIN" "$TAU_MAX")"

  echo ""
  echo "========== RUNNING ${SETTING}/${NAME} =========="
  echo "TASK=${TASK}"
  echo "FEWSHOT=${FEWSHOT}"
  echo "LIMIT=full"
  echo "GEN=${GEN}"
  echo "STEPS=${STEPS}"
  echo "BLOCK_MODE=${BLOCK_MODE}"
  echo "ALG=${ALG}"
  echo "TAU_MIN=${TAU_MIN}"
  echo "TAU_MAX=${TAU_MAX}"
  echo "MODEL_ARGS=${MODEL_ARGS}"
  echo ""

  CMD=(
    accelerate launch eval_llada.py
    --tasks "${TASK}"
    --model llada_dist
    --confirm_run_unsafe_code
    --model_args "${MODEL_ARGS}"
  )

  if [ "${FEWSHOT}" != "default" ]; then
    CMD+=(--num_fewshot "${FEWSHOT}")
  fi

  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  summarize_one "$LOG_FILE"
}

run_dapd () {
  SETTING="$1"
  TASK="$2"
  FEWSHOT="$3"
  GEN="$4"
  STEPS="$5"
  STAGED_TAU_MIN="$6"
  STAGED_TAU_MAX="$7"
  DIRECT_TAU_MIN="$8"
  DIRECT_TAU_MAX="$9"
  OUT_DIR="${10}"

  run_eval "$SETTING" "$TASK" "$FEWSHOT" "$GEN" "$STEPS" \
    "dapd_staged" "$STAGED_TAU_MIN" "$STAGED_TAU_MAX" "$OUT_DIR"

  if [ "$RUN_DIRECT" = "true" ]; then
    run_eval "$SETTING" "$TASK" "$FEWSHOT" "$GEN" "$STEPS" \
      "dapd_direct" "$DIRECT_TAU_MIN" "$DIRECT_TAU_MAX" "$OUT_DIR"
  fi
}

run_dapd "gsm8k_g256_s256_full" "gsm8k" "default" 256 256 \
  "0.01" "0.05" "0.005" "0.05" \
  "${BASE_OUT}/gsm8k_g256_s256_full"

run_dapd "gsm8k_g1024_s1024_full" "gsm8k" "default" 1024 1024 \
  "0.01" "0.05" "0.005" "0.05" \
  "${BASE_OUT}/gsm8k_g1024_s1024_full"

run_dapd "humaneval_g256_s256_full" "humaneval" "default" 256 256 \
  "0.01" "0.15" "0.01" "0.05" \
  "${BASE_OUT}/humaneval_g256_s256_full"

run_dapd "humaneval_g1024_s1024_full" "humaneval" "default" 1024 1024 \
  "0.01" "0.15" "0.01" "0.05" \
  "${BASE_OUT}/humaneval_g1024_s1024_full"

run_dapd "mbpp_g256_s256_full" "mbpp" "default" 256 256 \
  "0.01" "0.15" "0.01" "0.02" \
  "${BASE_OUT}/mbpp_g256_s256_full"

run_dapd "mbpp_g1024_s1024_full" "mbpp" "default" 1024 1024 \
  "0.01" "0.15" "0.01" "0.02" \
  "${BASE_OUT}/mbpp_g1024_s1024_full"

run_dapd "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "4" 256 256 \
  "0.01" "0.05" "0.005" "0.05" \
  "${BASE_OUT}/math_g256_s256_4shot_full"

run_dapd "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "4" 1024 1024 \
  "0.01" "0.05" "0.005" "0.05" \
  "${BASE_OUT}/math_g1024_s1024_4shot_full"

echo ""
echo "========== FINAL SUMMARY: all DAPD paper full =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
echo "========== END FINAL SUMMARY =========="
