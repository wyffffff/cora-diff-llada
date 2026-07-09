#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${ACTIVATE_CONDA:-true}" != "false" ]] && command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-cora}"
fi

if [ -n "${HF_ENDPOINT:-}" ]; then
  export HF_ENDPOINT
fi
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

TASK="${TASK:-score_non_greedy_robustness_math}"
NUM_FEWSHOT="${NUM_FEWSHOT:-4}"
OUT_DIR="${OUT_DIR:-output/other_tasks/math_g256_s256_4shot_full_eot_compare}"
mkdir -p "$OUT_DIR"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
GEN_LENGTH="${GEN_LENGTH:-256}"
STEPS="${STEPS:-256}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
REMASKING="${REMASKING:-low_confidence}"
CFG="${CFG:-0.0}"
LEARN2PD_ACCEPT_THRES="${LEARN2PD_ACCEPT_THRES:-0.96}"
CORA_EOT_CONFIDENCE_THRESHOLD="${CORA_EOT_CONFIDENCE_THRESHOLD:-0.80}"

COMMON_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG}"

ORIGINAL_ARGS="${COMMON_ARGS},method=original"
LEARN2PD_ARGS="${COMMON_ARGS},method=Learn2PD,accept_thres=${LEARN2PD_ACCEPT_THRES}"

make_cora_args() {
  local conf="$1"
  local eot="$2"
  local eot_thres="$3"

  echo "${COMMON_ARGS},method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=${conf},cora_accept_stability_steps=1,cora_eot=${eot},cora_eot_confidence_threshold=${eot_thres},cora_residual_accept=false,cora_residual_threshold=0.0,cora_drift_topk=1,cora_residual_beta=0.0,cora_residual_gamma=0.0"
}

CORA_CONF080_NOEOT_ARGS="$(make_cora_args 0.80 false "$CORA_EOT_CONFIDENCE_THRESHOLD")"
CORA_CONF080_EOT_ARGS="$(make_cora_args 0.80 true "$CORA_EOT_CONFIDENCE_THRESHOLD")"
CORA_CONF075_NOEOT_ARGS="$(make_cora_args 0.75 false "$CORA_EOT_CONFIDENCE_THRESHOLD")"
CORA_CONF075_EOT_ARGS="$(make_cora_args 0.75 true "$CORA_EOT_CONFIDENCE_THRESHOLD")"

summarize_one() {
  local name="$1"
  local log_file="${OUT_DIR}/${name}.log"

  echo ""
  echo "========== math_g256_s256_4shot_eot_compare/${name} SUMMARY =========="
  grep -E "Number of tokens|Generation time|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA residual accept tokens|CORA avg residual score|CORA EoT truncated samples|non_greedy_accuracy|exact_match" "$log_file" || true
  echo "========== END math_g256_s256_4shot_eot_compare/${name} SUMMARY =========="
  echo ""
}

run_eval() {
  local name="$1"
  local model_args="$2"

  echo ""
  echo "========== RUNNING math_g256_s256_4shot_eot_compare/${name} =========="
  echo "TASK=${TASK}"
  echo "NUM_FEWSHOT=${NUM_FEWSHOT}"
  echo "LIMIT=full"
  echo "MODEL_ARGS=${model_args}"
  echo ""

  accelerate launch eval_llada.py \
    --tasks "${TASK}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "${model_args}" \
    2>&1 | tee "${OUT_DIR}/${name}.log"

  summarize_one "$name"
}

run_eval "original" "${ORIGINAL_ARGS}"
run_eval "learn2pd" "${LEARN2PD_ARGS}"

run_eval "cora_conf080_stab1_noeot" "${CORA_CONF080_NOEOT_ARGS}"
run_eval "cora_conf080_stab1_eot" "${CORA_CONF080_EOT_ARGS}"

run_eval "cora_conf075_stab1_noeot" "${CORA_CONF075_NOEOT_ARGS}"
run_eval "cora_conf075_stab1_eot" "${CORA_CONF075_EOT_ARGS}"

echo ""
echo "========== FINAL SUMMARY: math_g256_s256_4shot_full_eot_compare =========="
grep -E "Number of tokens|Generation time|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA EoT truncated samples|non_greedy_accuracy|exact_match" "${OUT_DIR}"/*.log || true
echo "========== END FINAL SUMMARY: math_g256_s256_4shot_full_eot_compare =========="
