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
LIMIT="${LIMIT:-50}"
GEN_LENGTH="${GEN_LENGTH:-256}"
if [[ -z "${STEPS:-}" ]]; then
  if [[ "$GEN_LENGTH" == "1024" ]]; then
    STEPS=256
  else
    STEPS="$GEN_LENGTH"
  fi
fi
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
REMASKING="${REMASKING:-low_confidence}"
CFG="${CFG:-0.0}"

LEARN2PD_ACCEPT_THRES="${LEARN2PD_ACCEPT_THRES:-0.96}"
CORA_ACCEPT_CONFIDENCE_THRESHOLD="${CORA_ACCEPT_CONFIDENCE_THRESHOLD:-0.97}"
CORA_ACCEPT_STABILITY_STEPS="${CORA_ACCEPT_STABILITY_STEPS:-2}"
CORA_EOT_CONFIDENCE_THRESHOLD="${CORA_EOT_CONFIDENCE_THRESHOLD:-0.80}"

OUT_DIR="output/eot_pair/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

COMMON_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG}"
LEARN2PD_EOT_ARGS="${COMMON_ARGS},method=L2P+EoT,accept_thres=${LEARN2PD_ACCEPT_THRES}"
CORA_EOT_ARGS="${COMMON_ARGS},method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=${CORA_ACCEPT_CONFIDENCE_THRESHOLD},cora_accept_stability_steps=${CORA_ACCEPT_STABILITY_STEPS},cora_eot=true,cora_eot_confidence_threshold=${CORA_EOT_CONFIDENCE_THRESHOLD},cora_residual_accept=false,cora_residual_threshold=0.0,cora_drift_topk=1,cora_residual_beta=0.0,cora_residual_gamma=0.0,cora_persistence_temperature=2.0"

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

echo "Output directory: ${OUT_DIR}"
echo "TASK=${TASK}"
echo "LIMIT=${LIMIT}"
echo "GEN_LENGTH=${GEN_LENGTH}"
echo "STEPS=${STEPS}"
echo "BLOCK_LENGTH=${BLOCK_LENGTH}"

run_eval "learn2pd_eot" "$LEARN2PD_EOT_ARGS"
run_eval "cora_best_eot" "$CORA_EOT_ARGS"

echo "========== summary =========="
grep -E "Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA EoT truncated samples|exact_match|pass_at_1" "${OUT_DIR}"/*.log || true
python extract_log_table.py "$OUT_DIR" || true
echo "Logs: ${OUT_DIR}"
