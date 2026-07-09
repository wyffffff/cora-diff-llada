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
RANDOM_SAMPLE_SIZE="${RANDOM_SAMPLE_SIZE:-}"
RANDOM_SAMPLE_SEED="${RANDOM_SAMPLE_SEED:-2026}"
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

LEARN2PD_METHOD="${LEARN2PD_METHOD:-Learn2PD}"
LEARN2PD_EOT_METHOD="${LEARN2PD_EOT_METHOD:-L2P+EoT}"
LEARN2PD_ACCEPT_THRES="${LEARN2PD_ACCEPT_THRES:-0.96}"

CORA_ACCEPT_CONFIDENCE_THRESHOLD="${CORA_ACCEPT_CONFIDENCE_THRESHOLD:-0.97}"
CORA_ACCEPT_STABILITY_STEPS="${CORA_ACCEPT_STABILITY_STEPS:-2}"
CORA_EOT_CONFIDENCE_THRESHOLD="${CORA_EOT_CONFIDENCE_THRESHOLD:-0.80}"
CORA_PLUS_RESIDUAL_THRESHOLD="${CORA_PLUS_RESIDUAL_THRESHOLD:-${CORA_RESIDUAL_THRESHOLD:-0.5}}"
CORA_PLUS_DRIFT_TOPK="${CORA_PLUS_DRIFT_TOPK:-${CORA_DRIFT_TOPK:-8}}"
CORA_PLUS_RESIDUAL_BETA="${CORA_PLUS_RESIDUAL_BETA:-${CORA_RESIDUAL_BETA:-1.0}}"
CORA_PLUS_RESIDUAL_GAMMA="${CORA_PLUS_RESIDUAL_GAMMA:-${CORA_RESIDUAL_GAMMA:-1.0}}"
CORA_PLUS_PERSISTENCE_TEMPERATURE="${CORA_PLUS_PERSISTENCE_TEMPERATURE:-${CORA_PERSISTENCE_TEMPERATURE:-2.0}}"
RUN_CORA_PLUS="${RUN_CORA_PLUS:-true}"

if [[ -n "$RANDOM_SAMPLE_SIZE" && "$RANDOM_SAMPLE_SIZE" != "none" && "$RANDOM_SAMPLE_SIZE" != "None" ]]; then
  LIMIT_ARGS=()
  SAMPLE_TAG="random${RANDOM_SAMPLE_SIZE}_seed${RANDOM_SAMPLE_SEED}"
else
  LIMIT_ARGS=(--limit "$LIMIT")
  SAMPLE_TAG="limit${LIMIT}"
fi

OUT_DIR="output/eotp_compare/${TASK}_g${GEN_LENGTH}_s${STEPS}_${SAMPLE_TAG}"
mkdir -p "$OUT_DIR"

COMMON_ARGS="model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG}"
if [[ -n "$RANDOM_SAMPLE_SIZE" && "$RANDOM_SAMPLE_SIZE" != "none" && "$RANDOM_SAMPLE_SIZE" != "None" ]]; then
  COMMON_ARGS="${COMMON_ARGS},random_sample_size=${RANDOM_SAMPLE_SIZE},random_sample_seed=${RANDOM_SAMPLE_SEED}"
fi

ORIGINAL_ARGS="${COMMON_ARGS},method=original"
EOT_ARGS="${COMMON_ARGS},method=EoT"
LEARN2PD_ARGS="${COMMON_ARGS},method=${LEARN2PD_METHOD},accept_thres=${LEARN2PD_ACCEPT_THRES}"
LEARN2PD_EOT_ARGS="${COMMON_ARGS},method=${LEARN2PD_EOT_METHOD},accept_thres=${LEARN2PD_ACCEPT_THRES}"

CORA_BASE_ARGS="${COMMON_ARGS},method=CORA,cora_routing_mode=disabled,cora_num_groups=4,cora_core_ratio=0.5,cora_active_ratio=0.7,cora_active_ratio_final=0.45,cora_budget_schedule=cosine,cora_alpha=0.0,cora_dependency_topk=0,cora_order_channels=false,cora_channel_ordering=activation,cora_extra_commit=false,cora_fast_accept=true,cora_accept_confidence_threshold=${CORA_ACCEPT_CONFIDENCE_THRESHOLD},cora_accept_stability_steps=${CORA_ACCEPT_STABILITY_STEPS},cora_eot_confidence_threshold=${CORA_EOT_CONFIDENCE_THRESHOLD}"
CORA_BEST_BASE="${CORA_BASE_ARGS},cora_residual_accept=false,cora_residual_threshold=0.0,cora_drift_topk=1,cora_residual_beta=0.0,cora_residual_gamma=0.0,cora_persistence_temperature=${CORA_PLUS_PERSISTENCE_TEMPERATURE}"
CORA_PLUS_BASE="${CORA_BASE_ARGS},cora_residual_accept=true,cora_residual_threshold=${CORA_PLUS_RESIDUAL_THRESHOLD},cora_drift_topk=${CORA_PLUS_DRIFT_TOPK},cora_residual_beta=${CORA_PLUS_RESIDUAL_BETA},cora_residual_gamma=${CORA_PLUS_RESIDUAL_GAMMA},cora_persistence_temperature=${CORA_PLUS_PERSISTENCE_TEMPERATURE}"

run_eval() {
  local name="$1"
  local model_args="$2"
  local log_file="${OUT_DIR}/${name}.log"

  echo "========== ${name} =========="
  echo "model_args=${model_args}"
  accelerate launch eval_llada.py \
    --tasks "$TASK" \
    "${LIMIT_ARGS[@]}" \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --model_args "$model_args" \
    2>&1 | tee "$log_file"
}

echo "Output directory: ${OUT_DIR}"
echo "TASK=${TASK}"
if [[ ${#LIMIT_ARGS[@]} -eq 0 ]]; then
  echo "LIMIT=disabled"
  echo "RANDOM_SAMPLE_SIZE=${RANDOM_SAMPLE_SIZE}"
  echo "RANDOM_SAMPLE_SEED=${RANDOM_SAMPLE_SEED}"
else
  echo "LIMIT=${LIMIT}"
fi
echo "GEN_LENGTH=${GEN_LENGTH}"
echo "STEPS=${STEPS}"
echo "BLOCK_LENGTH=${BLOCK_LENGTH}"
echo "REMASKING=${REMASKING}"
echo "CFG=${CFG}"
echo "LEARN2PD_METHOD=${LEARN2PD_METHOD}"
echo "LEARN2PD_EOT_METHOD=${LEARN2PD_EOT_METHOD}"
echo "LEARN2PD_ACCEPT_THRES=${LEARN2PD_ACCEPT_THRES}"
echo "CORA confidence=${CORA_ACCEPT_CONFIDENCE_THRESHOLD} stability=${CORA_ACCEPT_STABILITY_STEPS} eot_threshold=${CORA_EOT_CONFIDENCE_THRESHOLD}"
echo "RUN_CORA_PLUS=${RUN_CORA_PLUS}"

run_eval "original" "$ORIGINAL_ARGS"
run_eval "eot" "$EOT_ARGS"
run_eval "learn2pd" "$LEARN2PD_ARGS"
run_eval "learn2pd_eot" "$LEARN2PD_EOT_ARGS"
run_eval "cora_best_noeot" "${CORA_BEST_BASE},cora_eot=false"
run_eval "cora_best_eot" "${CORA_BEST_BASE},cora_eot=true"

if [[ "$RUN_CORA_PLUS" != "false" && "$RUN_CORA_PLUS" != "0" && "$RUN_CORA_PLUS" != "no" ]]; then
  run_eval "cora_plus_noeot" "${CORA_PLUS_BASE},cora_eot=false"
  run_eval "cora_plus_eot" "${CORA_PLUS_BASE},cora_eot=true"
fi

echo "========== summary =========="
grep -E "Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA residual accept tokens|CORA avg residual score|CORA EoT truncated samples|exact_match|pass_at_1" "${OUT_DIR}"/*.log || true
python extract_log_table.py "$OUT_DIR" || true
echo "Logs: ${OUT_DIR}"
