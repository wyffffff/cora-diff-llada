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
STEPS="${STEPS:-$GEN_LENGTH}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
REMASKING="${REMASKING:-low_confidence}"
CFG="${CFG:-0.0}"

PROPHET_CONSTRAINTS_TEXT="${PROPHET_CONSTRAINTS_TEXT:-}"
PROPHET_ANSWER_START="${PROPHET_ANSWER_START:-}"
PROPHET_ANSWER_LENGTH="${PROPHET_ANSWER_LENGTH:-5}"
PROPHET_EARLY_THRESHOLD="${PROPHET_EARLY_THRESHOLD:-7.5}"
PROPHET_MID_THRESHOLD="${PROPHET_MID_THRESHOLD:-5.0}"
PROPHET_LATE_THRESHOLD="${PROPHET_LATE_THRESHOLD:-2.5}"

KLASS_CONF_THRESHOLD="${KLASS_CONF_THRESHOLD:-0.9}"
KLASS_KL_THRESHOLD="${KLASS_KL_THRESHOLD:-0.01}"
KLASS_HISTORY_LENGTH="${KLASS_HISTORY_LENGTH:-2}"
KLASS_UNMASK_STRATEGY="${KLASS_UNMASK_STRATEGY:-all}"
KLASS_CONFIDENCE_METRIC="${KLASS_CONFIDENCE_METRIC:-prob}"
KLASS_ALG="${KLASS_ALG:-klass}"

DAPD_LAYER_RATIO="${DAPD_LAYER_RATIO:-0.3}"
DAPD_TAU_MIN="${DAPD_TAU_MIN:-0.01}"
DAPD_TAU_MAX="${DAPD_TAU_MAX:-0.15}"

OUT_DIR="output/prophet_klass_dapd_compare/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

COMMON_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG}"
ORIGINAL_ARGS="${COMMON_ARGS},method=original"
PROPHET_ARGS="${COMMON_ARGS},method=Prophet,prophet_constraints_text=${PROPHET_CONSTRAINTS_TEXT},prophet_answer_length=${PROPHET_ANSWER_LENGTH},prophet_early_threshold=${PROPHET_EARLY_THRESHOLD},prophet_mid_threshold=${PROPHET_MID_THRESHOLD},prophet_late_threshold=${PROPHET_LATE_THRESHOLD}"
if [ -n "$PROPHET_ANSWER_START" ]; then
  PROPHET_ARGS="${PROPHET_ARGS},prophet_answer_start=${PROPHET_ANSWER_START}"
fi
KLASS_ARGS="${COMMON_ARGS},method=KLASS,klass_alg=${KLASS_ALG},klass_conf_threshold=${KLASS_CONF_THRESHOLD},klass_kl_threshold=${KLASS_KL_THRESHOLD},klass_history_length=${KLASS_HISTORY_LENGTH},klass_unmask_strategy=${KLASS_UNMASK_STRATEGY},klass_confidence_metric=${KLASS_CONFIDENCE_METRIC}"
DAPD_STAGED_ARGS="${COMMON_ARGS},method=DAPD,dapd_alg=dapd_staged,dapd_layer_ratio=${DAPD_LAYER_RATIO},dapd_tau_min=${DAPD_TAU_MIN},dapd_tau_max=${DAPD_TAU_MAX}"
DAPD_DIRECT_ARGS="${COMMON_ARGS},method=DAPD,dapd_alg=dapd_direct,dapd_layer_ratio=${DAPD_LAYER_RATIO},dapd_tau_min=${DAPD_TAU_MIN},dapd_tau_max=${DAPD_TAU_MAX}"

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

run_eval "original" "$ORIGINAL_ARGS"
run_eval "prophet" "$PROPHET_ARGS"
run_eval "klass" "$KLASS_ARGS"
run_eval "dapd_staged" "$DAPD_STAGED_ARGS"
run_eval "dapd_direct" "$DAPD_DIRECT_ARGS"

echo "========== summary =========="
grep -E "Tokens per second|Prophet|KLASS|DAPD|exact_match|pass_at_1" "${OUT_DIR}"/*.log || true
echo "Logs: ${OUT_DIR}"
