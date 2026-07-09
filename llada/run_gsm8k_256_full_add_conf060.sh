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

TASK="gsm8k"
OUT_DIR="output/pareto/gsm8k_g256_s256_full_noeot_stab012"
mkdir -p "$OUT_DIR"

COMMON_ARGS="model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=256,steps=256,block_length=32,remasking=low_confidence,cfg=0.0"

make_cora_args () {
  CONF="$1"
  STAB="$2"

  echo "${COMMON_ARGS},method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=${CONF},cora_accept_stability_steps=${STAB},cora_eot=false,cora_eot_confidence_threshold=1.0,cora_residual_accept=false,cora_residual_threshold=0.0,cora_drift_topk=1,cora_residual_beta=0.0,cora_residual_gamma=0.0"
}

summarize_one () {
  NAME="$1"
  LOG_FILE="${OUT_DIR}/${NAME}.log"

  echo ""
  echo "========== ${NAME} SUMMARY =========="
  grep -E "Number of tokens|Generation time|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|flexible-extract|strict-match|exact_match" "$LOG_FILE" || true
  echo "========== END ${NAME} SUMMARY =========="
}

run_eval () {
  NAME="$1"
  MODEL_ARGS="$2"
  LOG_FILE="${OUT_DIR}/${NAME}.log"

  if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "========== SKIP ${NAME}: log already exists =========="
    summarize_one "$NAME"
    return 0
  fi

  echo ""
  echo "========== RUNNING ${NAME} =========="
  echo "TASK=${TASK}"
  echo "LIMIT=full"
  echo "MODEL_ARGS=${MODEL_ARGS}"

  accelerate launch eval_llada.py \
    --tasks "${TASK}" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "${MODEL_ARGS}" \
    2>&1 | tee "$LOG_FILE"

  summarize_one "$NAME"
}

run_eval "cora_conf060_stab0_noeot" "$(make_cora_args 0.60 0)"
run_eval "cora_conf060_stab1_noeot" "$(make_cora_args 0.60 1)"
run_eval "cora_conf060_stab2_noeot" "$(make_cora_args 0.60 2)"

echo ""
echo "========== FINAL SUMMARY: conf060 added =========="
grep -E "Number of tokens|Generation time|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|flexible-extract|strict-match|exact_match" "${OUT_DIR}"/cora_conf060_*.log || true
echo "========== END FINAL SUMMARY =========="
