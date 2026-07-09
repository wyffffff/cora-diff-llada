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
LIMIT="${LIMIT:-5}"
GEN_LENGTH="${GEN_LENGTH:-128}"
STEPS="${STEPS:-$GEN_LENGTH}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"

OUT_DIR="output/diagnostics/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

run_eval() {
  local name="$1"
  local model_args="$2"
  local log_file="${OUT_DIR}/${name}.log"
  echo "========== ${name} =========="
  echo "model_args=${model_args}"
  accelerate launch eval_llada.py \
    --tasks "$TASK" \
    --limit "$LIMIT" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "$model_args" \
    2>&1 | tee "$log_file"
}

COMMON="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH}"

run_eval "original" \
  "${COMMON},method=original"

# CORA outer loop and statistics machinery, but native dense MLP. This isolates
# generate_cora/control overhead from routed FFN overhead.
run_eval "cora_control_native_mlp" \
  "${COMMON},method=CORA,cora_routing_mode=disabled,cora_order_channels=false,cora_extra_commit=false,cora_alpha=0.0,cora_dependency_topk=0"

# Dense MLP plus Learn2PD/EoTP-style step reduction. This is the next candidate
# for real wall-clock speedup because it reduces actual denoising forwards.
run_eval "cora_step_native" \
  "${COMMON},method=CORA,cora_routing_mode=disabled,cora_order_channels=false,cora_extra_commit=false,cora_alpha=0.0,cora_dependency_topk=0,cora_fast_accept=true,cora_accept_confidence_threshold=0.90,cora_accept_stability_steps=1,cora_eot=true,cora_eot_confidence_threshold=0.50"

# Routed FFN with all refinement groups active. This isolates the dynamic
# grouped-MLP implementation overhead when no compute is saved.
run_eval "cora_dense_routed" \
  "${COMMON},method=CORA,cora_routing_mode=dense_routed,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=1.0,cora_order_channels=false,cora_extra_commit=false,cora_alpha=0.0,cora_dependency_topk=0"

# Aggressive adaptive routing. If this is still slower than original, the
# current fine-grained routed FFN is not GPU-efficient enough for wall-clock speed.
run_eval "cora_aggressive" \
  "${COMMON},method=CORA,cora_routing_mode=adaptive,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.25,cora_active_ratio_final=0.10,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_extra_commit=false"

echo "Diagnostics complete. Logs: ${OUT_DIR}"
