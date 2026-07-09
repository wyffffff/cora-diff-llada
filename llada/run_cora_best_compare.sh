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
GEN_LENGTH="${GEN_LENGTH:-128}"
STEPS="${STEPS:-$GEN_LENGTH}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"

CORA_ACCEPT_CONFIDENCE_THRESHOLD="${CORA_ACCEPT_CONFIDENCE_THRESHOLD:-0.97}"
CORA_ACCEPT_STABILITY_STEPS="${CORA_ACCEPT_STABILITY_STEPS:-2}"
CORA_EOT="${CORA_EOT:-false}"
CORA_EOT_CONFIDENCE_THRESHOLD="${CORA_EOT_CONFIDENCE_THRESHOLD:-0.80}"
CORA_RESIDUAL_ACCEPT="${CORA_RESIDUAL_ACCEPT:-false}"
CORA_RESIDUAL_THRESHOLD="${CORA_RESIDUAL_THRESHOLD:-0.0}"
CORA_DRIFT_TOPK="${CORA_DRIFT_TOPK:-8}"
CORA_RESIDUAL_BETA="${CORA_RESIDUAL_BETA:-1.0}"
CORA_RESIDUAL_GAMMA="${CORA_RESIDUAL_GAMMA:-1.0}"
CORA_PERSISTENCE_TEMPERATURE="${CORA_PERSISTENCE_TEMPERATURE:-2.0}"

OUT_DIR="output/best_compare/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

ORIGINAL_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},method=original"
CORA_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=${CORA_ACCEPT_CONFIDENCE_THRESHOLD},cora_accept_stability_steps=${CORA_ACCEPT_STABILITY_STEPS},cora_eot=${CORA_EOT},cora_eot_confidence_threshold=${CORA_EOT_CONFIDENCE_THRESHOLD},cora_residual_accept=${CORA_RESIDUAL_ACCEPT},cora_residual_threshold=${CORA_RESIDUAL_THRESHOLD},cora_drift_topk=${CORA_DRIFT_TOPK},cora_residual_beta=${CORA_RESIDUAL_BETA},cora_residual_gamma=${CORA_RESIDUAL_GAMMA},cora_persistence_temperature=${CORA_PERSISTENCE_TEMPERATURE}"

echo "Output directory: ${OUT_DIR}"
echo "TASK=${TASK}"
echo "LIMIT=${LIMIT}"
echo "GEN_LENGTH=${GEN_LENGTH}"
echo "STEPS=${STEPS}"
echo "BLOCK_LENGTH=${BLOCK_LENGTH}"

echo "========== original =========="
echo "model_args=${ORIGINAL_ARGS}"
accelerate launch eval_llada.py \
  --tasks "$TASK" \
  --limit "$LIMIT" \
  --model llada_dist \
  --confirm_run_unsafe_code \
  --model_args "$ORIGINAL_ARGS" \
  2>&1 | tee "${OUT_DIR}/original.log"

echo "========== cora_step_best =========="
echo "model_args=${CORA_ARGS}"
accelerate launch eval_llada.py \
  --tasks "$TASK" \
  --limit "$LIMIT" \
  --model llada_dist \
  --confirm_run_unsafe_code \
  --model_args "$CORA_ARGS" \
  2>&1 | tee "${OUT_DIR}/cora_step_best.log"

echo "========== summary =========="
grep -E "Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA residual accept tokens|CORA avg residual score|CORA EoT truncated samples|exact_match" "${OUT_DIR}"/*.log || true
echo "Logs: ${OUT_DIR}"
