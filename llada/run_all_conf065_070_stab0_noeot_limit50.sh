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

BASE_OUT="output/ablations/all_tasks_conf065_070_stab0_noeot_limit50"
mkdir -p "$BASE_OUT"

make_cora_args () {
  GEN="$1"
  STEPS="$2"
  CONF="$3"

  echo "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=${GEN},steps=${STEPS},block_length=32,remasking=low_confidence,cfg=0.0,method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=${CONF},cora_accept_stability_steps=0,cora_eot=false,cora_eot_confidence_threshold=1.0,cora_residual_accept=false,cora_residual_threshold=0.0,cora_drift_topk=1,cora_residual_beta=0.0,cora_residual_gamma=0.0"
}

summarize_one () {
  LOG_FILE="$1"

  echo ""
  echo "========== SUMMARY: ${LOG_FILE} =========="
  grep -E "Number of tokens|Generation time|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|flexible-extract|strict-match|exact_match|pass_at_1|non_greedy_accuracy" "$LOG_FILE" || true
  echo "========== END SUMMARY =========="
}

run_eval () {
  SETTING="$1"
  TASK="$2"
  FEWSHOT="$3"
  GEN="$4"
  STEPS="$5"
  CONF="$6"
  OUT_DIR="$7"

  CONF_TAG="${CONF/./}"
  NAME="cora_conf${CONF_TAG}_stab0_noeot_limit50"
  LOG_FILE="${OUT_DIR}/${NAME}.log"

  mkdir -p "$OUT_DIR"

  if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "========== SKIP ${SETTING}/${NAME}: log already exists =========="
    summarize_one "$LOG_FILE"
    return 0
  fi

  MODEL_ARGS="$(make_cora_args "$GEN" "$STEPS" "$CONF")"

  echo ""
  echo "========== RUNNING ${SETTING}/${NAME} =========="
  echo "TASK=${TASK}"
  echo "FEWSHOT=${FEWSHOT}"
  echo "LIMIT=50"
  echo "GEN=${GEN}"
  echo "STEPS=${STEPS}"
  echo "MODEL_ARGS=${MODEL_ARGS}"
  echo ""

  CMD=(
    accelerate launch eval_llada.py
    --tasks "${TASK}"
    --limit 50
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

run_pair () {
  SETTING="$1"
  TASK="$2"
  FEWSHOT="$3"
  GEN="$4"
  STEPS="$5"
  OUT_DIR="$6"

  run_eval "$SETTING" "$TASK" "$FEWSHOT" "$GEN" "$STEPS" "0.65" "$OUT_DIR"
  run_eval "$SETTING" "$TASK" "$FEWSHOT" "$GEN" "$STEPS" "0.70" "$OUT_DIR"
}

run_pair "gsm8k_g256_s256_limit50" "gsm8k" "default" 256 256 \
  "${BASE_OUT}/gsm8k_g256_s256_limit50"

run_pair "gsm8k_g1024_s1024_limit50" "gsm8k" "default" 1024 1024 \
  "${BASE_OUT}/gsm8k_g1024_s1024_limit50"

run_pair "humaneval_g256_s256_limit50" "humaneval" "default" 256 256 \
  "${BASE_OUT}/humaneval_g256_s256_limit50"

run_pair "humaneval_g1024_s1024_limit50" "humaneval" "default" 1024 1024 \
  "${BASE_OUT}/humaneval_g1024_s1024_limit50"

run_pair "mbpp_g256_s256_limit50" "mbpp" "default" 256 256 \
  "${BASE_OUT}/mbpp_g256_s256_limit50"

run_pair "mbpp_g1024_s1024_limit50" "mbpp" "default" 1024 1024 \
  "${BASE_OUT}/mbpp_g1024_s1024_limit50"

run_pair "math_g256_s256_4shot_limit50" "score_non_greedy_robustness_math" "4" 256 256 \
  "${BASE_OUT}/math_g256_s256_4shot_limit50"

echo ""
echo "========== FINAL SUMMARY: all conf065/conf070 stab0 no-EOT limit50 =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|flexible-extract|strict-match|exact_match|pass_at_1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
echo "========== END FINAL SUMMARY =========="
