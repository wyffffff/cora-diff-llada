#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -n "${CONDA_ENV:-}" ] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "cora"; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cora
fi

if [ -n "${HF_ENDPOINT:-}" ]; then
  export HF_ENDPOINT
fi
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export KLASS_PROB_DTYPE="${KLASS_PROB_DTYPE:-float32}"
export KLASS_KL_VOCAB_CHUNK="${KLASS_KL_VOCAB_CHUNK:-8192}"
export DAPD_ATTENTION_HEAD_CHUNK="${DAPD_ATTENTION_HEAD_CHUNK:-2}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
TASK="${TASK:-gsm8k}"
LIMIT="${LIMIT:-1}"
GEN_LENGTH="${GEN_LENGTH:-64}"
STEPS="${STEPS:-64}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
OUT_DIR="${OUT_DIR:-output/smoke_open_source}"
METHODS="${METHODS:-original CORA Prophet KLASS DAPD}"

mkdir -p "$OUT_DIR"

common_args () {
  echo "model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=low_confidence,cfg=0.0"
}

model_args_for () {
  local method="$1"
  local common
  common="$(common_args)"

  case "$method" in
    original)
      echo "${common},method=original"
      ;;
    CORA)
      echo "${common},method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=0.97,cora_accept_stability_steps=2,cora_eot=false,cora_residual_accept=false"
      ;;
    Prophet)
      echo "${common},method=Prophet,prophet_answer_start=48,prophet_answer_length=4,prophet_early_threshold=7.5,prophet_mid_threshold=5.0,prophet_late_threshold=2.5"
      ;;
    KLASS)
      echo "${common},method=KLASS,klass_alg=klass,klass_conf_threshold=0.6,klass_kl_threshold=0.015,klass_history_length=2,klass_unmask_strategy=all,klass_confidence_metric=prob"
      ;;
    DAPD)
      echo "${common},method=DAPD,dapd_alg=dapd_staged,dapd_layer_ratio=0.3,dapd_tau_min=0.01,dapd_tau_max=0.05"
      ;;
    *)
      echo "Unknown method: ${method}" >&2
      return 1
      ;;
  esac
}

echo "========== Open-source smoke run =========="
echo "Model: ${MODEL_PATH}"
echo "Task: ${TASK}"
echo "Limit: ${LIMIT}"
echo "Generation: ${GEN_LENGTH}/${STEPS}, block=${BLOCK_LENGTH}"
echo "Methods: ${METHODS}"
echo "Output: ${OUT_DIR}"

for method in $METHODS; do
  log_file="${OUT_DIR}/${method}_${TASK}_limit${LIMIT}_g${GEN_LENGTH}_s${STEPS}.log"
  args="$(model_args_for "$method")"

  echo ""
  echo "========== RUNNING ${method} =========="
  echo "MODEL_ARGS=${args}"

  accelerate launch eval_llada.py \
    --tasks "$TASK" \
    --limit "$LIMIT" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "$args" \
    2>&1 | tee "$log_file"
done

echo ""
echo "========== Smoke summary =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|CORA|Prophet|KLASS|DAPD|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$OUT_DIR"/*.log || true
echo "Logs: ${OUT_DIR}"
