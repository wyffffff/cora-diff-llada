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

BASE_OUT="output/ablations/klass_besthp_fair_block32_full"
mkdir -p "$BASE_OUT"

MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
BLOCK_LENGTH=32
REMASKING="low_confidence"
CFG="0.0"

summarize_one () {
  LOG_FILE="$1"
  echo ""
  echo "========== SUMMARY: ${LOG_FILE} =========="
  grep -E "Number of tokens|Generation time|Tokens per second|KLASS actual denoising step ratio|KLASS ready tokens|KLASS fallback tokens|KLASS avg KL|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$LOG_FILE" || true
  echo "========== END SUMMARY =========="
}

is_complete () {
  LOG_FILE="$1"
  [ -f "$LOG_FILE" ] || return 1
  grep -Eq "Number of tokens|Generation time|Tokens per second" "$LOG_FILE" || return 1
  grep -Eq "exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$LOG_FILE" || return 1
  ! grep -Eq "Traceback|OutOfMemoryError|CUDA out of memory|returned non-zero exit status|Task.*not found|KeyError" "$LOG_FILE"
}

make_klass_args () {
  GEN="$1"
  STEPS="$2"
  CONF="$3"
  KL="$4"

  echo "model_path=${MODEL_PATH},gen_length=${GEN},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG},method=KLASS,klass_alg=klass,klass_conf_threshold=${CONF},klass_kl_threshold=${KL},klass_history_length=2,klass_unmask_strategy=all,klass_confidence_metric=prob"
}

run_eval () {
  SETTING="$1"
  TASK="$2"
  FEWSHOT="$3"
  GEN="$4"
  STEPS="$5"
  CONF="$6"
  KL="$7"

  CONF_TAG="${CONF/./}"
  KL_TAG="${KL/./}"
  OUT_DIR="${BASE_OUT}/${SETTING}"
  LOG_FILE="${OUT_DIR}/klass_conf${CONF_TAG}_kl${KL_TAG}.log"

  mkdir -p "$OUT_DIR"

  if is_complete "$LOG_FILE"; then
    echo ""
    echo "========== SKIP ${SETTING}: completed =========="
    summarize_one "$LOG_FILE"
    return 0
  fi

  if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.failed_$(date +%Y%m%d_%H%M%S)"
  fi

  MODEL_ARGS="$(make_klass_args "$GEN" "$STEPS" "$CONF" "$KL")"

  echo ""
  echo "========== RUNNING ${SETTING} =========="
  echo "TASK=${TASK}"
  echo "FEWSHOT=${FEWSHOT}"
  echo "GEN=${GEN}"
  echo "STEPS=${STEPS}"
  echo "BLOCK_LENGTH=${BLOCK_LENGTH}"
  echo "CONF=${CONF}"
  echo "KL=${KL}"
  echo "MODEL_ARGS=${MODEL_ARGS}"

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

# KLASS paper best thresholds for LLaDA:
# GSM8K:    conf=0.6, kl=0.015
# MATH:     conf=0.6, kl=0.01
# HumanEval conf=0.9, kl=0.01
# MBPP:     conf=0.7, kl=0.01

run_eval "gsm8k_g256_s256_full" "gsm8k" "default" 256 256 "0.6" "0.015"
run_eval "gsm8k_g1024_s1024_full" "gsm8k" "default" 1024 1024 "0.6" "0.015"

run_eval "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "4" 256 256 "0.6" "0.01"
run_eval "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "4" 1024 1024 "0.6" "0.01"

run_eval "humaneval_g256_s256_full" "humaneval" "default" 256 256 "0.9" "0.01"
run_eval "humaneval_g1024_s1024_full" "humaneval" "default" 1024 1024 "0.9" "0.01"

run_eval "mbpp_g256_s256_full" "mbpp" "default" 256 256 "0.7" "0.01"
run_eval "mbpp_g1024_s1024_full" "mbpp" "default" 1024 1024 "0.7" "0.01"

echo ""
echo "========== FINAL SUMMARY: KLASS best hparams fair block32 =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|KLASS actual denoising step ratio|KLASS ready tokens|KLASS fallback tokens|KLASS avg KL|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
python extract_log_table.py "$BASE_OUT" || true
echo "Logs: ${BASE_OUT}"
