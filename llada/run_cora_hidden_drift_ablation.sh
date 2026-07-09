#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

TASK="${TASK:-gsm8k}"
MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
LIMIT="${LIMIT:-20}"
GEN_LENGTH="${GEN_LENGTH:-128}"
STEPS="${STEPS:-$GEN_LENGTH}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"

CORA_ACTIVE_RATIO="${CORA_ACTIVE_RATIO:-0.25}"
CORA_ACTIVE_RATIO_FINAL="${CORA_ACTIVE_RATIO_FINAL:-0.10}"
CORA_CORE_RATIO="${CORA_CORE_RATIO:-0.5}"
CORA_NUM_GROUPS="${CORA_NUM_GROUPS:-4}"
CORA_BETA="${CORA_BETA:-1.0}"

OUT_DIR="output/hidden_drift_ablation/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

run_eval() {
  local name="$1"
  local gamma="$2"
  local log_file="${OUT_DIR}/${name}.log"
  local model_args

  model_args="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},method=CORA,cora_routing_mode=adaptive,cora_num_groups=${CORA_NUM_GROUPS},cora_core_ratio=${CORA_CORE_RATIO},cora_active_ratio=${CORA_ACTIVE_RATIO},cora_active_ratio_final=${CORA_ACTIVE_RATIO_FINAL},cora_budget_schedule=cosine,cora_alpha=0.0,cora_beta=${CORA_BETA},cora_gamma=${gamma},cora_dependency_topk=0,cora_order_channels=false,cora_extra_commit=false,cora_fast_accept=false,cora_eot=false"

  echo "========== ${name} =========="
  echo "gamma=${gamma}"
  echo "model_args=${model_args}"

  accelerate launch eval_llada.py \
    --tasks "$TASK" \
    --limit "$LIMIT" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "$model_args" \
    2>&1 | tee "$log_file"
}

run_eval "gamma0_no_hidden_drift" "0.0"
run_eval "gamma1_hidden_drift" "1.0"
run_eval "gamma2_stronger_hidden_drift" "2.0"

echo "========== summary =========="
grep -E "Tokens per second|CORA avg refinement ratio|CORA estimated activated FFN ratio|exact_match" "${OUT_DIR}"/*.log || true
echo "Logs: ${OUT_DIR}"
