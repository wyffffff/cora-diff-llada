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

# Only used by DAPD at the end to avoid OOM on 1024-token settings.
export DAPD_ATTENTION_HEAD_CHUNK="${DAPD_ATTENTION_HEAD_CHUNK:-4}"

BASE_OUT="output/ablations/pk_first_then_three_remaining_full"
mkdir -p "$BASE_OUT"

MODEL_PATH="GSAI-ML/LLaDA-8B-Instruct"
BLOCK_LENGTH=32
REMASKING="low_confidence"
CFG="0.0"

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

common_args () {
  local gen="$1"
  local steps="$2"
  echo "model_path=${MODEL_PATH},gen_length=${gen},steps=${steps},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG}"
}

method_args () {
  local method="$1"
  local gen="$2"
  local steps="$3"
  local tau_min="$4"
  local tau_max="$5"
  local common
  common="$(common_args "$gen" "$steps")"

  case "$method" in
    prophet)
      echo "${common},method=Prophet,prophet_answer_length=${PROPHET_ANSWER_LENGTH},prophet_early_threshold=${PROPHET_EARLY_THRESHOLD},prophet_mid_threshold=${PROPHET_MID_THRESHOLD},prophet_late_threshold=${PROPHET_LATE_THRESHOLD}"
      ;;
    klass)
      echo "${common},method=KLASS,klass_alg=${KLASS_ALG},klass_conf_threshold=${KLASS_CONF_THRESHOLD},klass_kl_threshold=${KLASS_KL_THRESHOLD},klass_history_length=${KLASS_HISTORY_LENGTH},klass_unmask_strategy=${KLASS_UNMASK_STRATEGY},klass_confidence_metric=${KLASS_CONFIDENCE_METRIC}"
      ;;
    dapd)
      echo "${common},method=DAPD,dapd_alg=dapd_staged,dapd_layer_ratio=0.3,dapd_tau_min=${tau_min},dapd_tau_max=${tau_max}"
      ;;
    *)
      echo "unknown method: $method" >&2
      return 1
      ;;
  esac
}

summarize_one () {
  local log_file="$1"
  echo ""
  echo "========== SUMMARY: ${log_file} =========="
  grep -E "Number of tokens|Generation time|Tokens per second|Prophet|KLASS|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$log_file" || true
  echo "========== END SUMMARY =========="
}

is_complete () {
  local log_file="$1"
  [ -f "$log_file" ] || return 1
  grep -Eq "Number of tokens|Generation time|Tokens per second" "$log_file" || return 1
  grep -Eq "exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$log_file" || return 1
  ! grep -Eq "Traceback|OutOfMemoryError|CUDA out of memory|returned non-zero exit status" "$log_file"
}

run_eval () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local gen="$4"
  local steps="$5"
  local tau_min="$6"
  local tau_max="$7"
  local method="$8"

  local out_dir="${BASE_OUT}/${setting}"
  local log_file="${out_dir}/${method}.log"
  mkdir -p "$out_dir"

  if is_complete "$log_file"; then
    echo ""
    echo "========== SKIP ${setting}/${method}: completed =========="
    summarize_one "$log_file"
    return 0
  fi

  if [ -f "$log_file" ]; then
    mv "$log_file" "${log_file}.failed_$(date +%Y%m%d_%H%M%S)"
  fi

  local model_args
  model_args="$(method_args "$method" "$gen" "$steps" "$tau_min" "$tau_max")"

  echo ""
  echo "========== RUNNING ${setting}/${method} =========="
  echo "TASK=${task}"
  echo "FEWSHOT=${fewshot}"
  echo "GEN=${gen}"
  echo "STEPS=${steps}"
  echo "MODEL_ARGS=${model_args}"

  local cmd=(
    accelerate launch eval_llada.py
    --tasks "$task"
    --model llada_dist
    --confirm_run_unsafe_code
    --model_args "$model_args"
  )

  if [ "$fewshot" != "default" ]; then
    cmd+=(--num_fewshot "$fewshot")
  fi

  "${cmd[@]}" 2>&1 | tee "$log_file"
  summarize_one "$log_file"
}

run_methods () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local gen="$4"
  local steps="$5"
  local tau_min="$6"
  local tau_max="$7"
  shift 7

  local method
  for method in "$@"; do
    run_eval "$setting" "$task" "$fewshot" "$gen" "$steps" "$tau_min" "$tau_max" "$method"
  done
}

echo "========== PHASE 1: Prophet/KLASS for settings where DAPD already completed =========="
run_methods "gsm8k_g256_s256_full" "gsm8k" "default" 256 256 "0.01" "0.05" prophet klass
run_methods "gsm8k_g1024_s1024_full" "gsm8k" "default" 1024 1024 "0.01" "0.05" prophet klass
run_methods "humaneval_g256_s256_full" "humaneval" "default" 256 256 "0.01" "0.15" prophet klass
run_methods "humaneval_g1024_s1024_full" "humaneval" "default" 1024 1024 "0.01" "0.15" prophet klass
run_methods "mbpp_g256_s256_full" "mbpp" "default" 256 256 "0.01" "0.15" prophet klass

echo ""
echo "========== PHASE 2: remaining settings, run all three methods =========="
run_methods "mbpp_g1024_s1024_full" "mbpp" "default" 1024 1024 "0.01" "0.15" prophet klass dapd
run_methods "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "4" 256 256 "0.01" "0.05" prophet klass dapd
run_methods "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "4" 1024 1024 "0.01" "0.05" prophet klass dapd

echo ""
echo "========== FINAL SUMMARY =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|Prophet|KLASS|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
echo "Logs: ${BASE_OUT}"
