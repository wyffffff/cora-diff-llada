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

# Keeps DAPD's attention dependency calculation equivalent, but computes heads
# in chunks so long MBPP/MATH prompts do not require one huge QK allocation.
export DAPD_ATTENTION_HEAD_CHUNK="${DAPD_ATTENTION_HEAD_CHUNK:-4}"

BASE_OUT="output/ablations/all_tasks_dapd_fair_block32_full_resume_headchunk${DAPD_ATTENTION_HEAD_CHUNK}"
mkdir -p "$BASE_OUT"

make_dapd_args () {
  GEN="$1"
  STEPS="$2"
  TAU_MIN="$3"
  TAU_MAX="$4"

  echo "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=${GEN},steps=${STEPS},block_length=32,remasking=low_confidence,cfg=0.0,method=DAPD,dapd_alg=dapd_staged,dapd_layer_ratio=0.3,dapd_tau_min=${TAU_MIN},dapd_tau_max=${TAU_MAX}"
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
  TAU_MIN="$6"
  TAU_MAX="$7"
  OUT_DIR="$8"

  TAU_MIN_TAG="${TAU_MIN/./}"
  TAU_MAX_TAG="${TAU_MAX/./}"
  NAME="dapd_staged_tau${TAU_MIN_TAG}_${TAU_MAX_TAG}_full_headchunk${DAPD_ATTENTION_HEAD_CHUNK}"
  LOG_FILE="${OUT_DIR}/${NAME}.log"

  mkdir -p "$OUT_DIR"

  if [ -f "$LOG_FILE" ] && grep -Eq "Number of tokens|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$LOG_FILE"; then
    echo ""
    echo "========== SKIP ${SETTING}/${NAME}: completed log already exists =========="
    summarize_one "$LOG_FILE"
    return 0
  fi

  if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.failed_$(date +%Y%m%d_%H%M%S)"
  fi

  MODEL_ARGS="$(make_dapd_args "$GEN" "$STEPS" "$TAU_MIN" "$TAU_MAX")"

  echo ""
  echo "========== RUNNING ${SETTING}/${NAME} =========="
  echo "TASK=${TASK}"
  echo "FEWSHOT=${FEWSHOT}"
  echo "GEN=${GEN}"
  echo "STEPS=${STEPS}"
  echo "DAPD_ATTENTION_HEAD_CHUNK=${DAPD_ATTENTION_HEAD_CHUNK}"
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

run_eval "mbpp_g1024_s1024_full" "mbpp" "default" 1024 1024 "0.01" "0.15" \
  "${BASE_OUT}/mbpp_g1024_s1024_full"

run_eval "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "4" 256 256 "0.01" "0.05" \
  "${BASE_OUT}/math_g256_s256_4shot_full"

run_eval "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "4" 1024 1024 "0.01" "0.05" \
  "${BASE_OUT}/math_g1024_s1024_4shot_full"

echo ""
echo "========== FINAL SUMMARY: DAPD fair resume headchunk${DAPD_ATTENTION_HEAD_CHUNK} =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
echo "========== END FINAL SUMMARY =========="
