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
export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DAPD_ATTENTION_HEAD_CHUNK="${DAPD_ATTENTION_HEAD_CHUNK:-4}"

BASE_OUT="${BASE_OUT:-output/ablations/best_hparams_fair_block32_full_prophet_answerstart200}"
mkdir -p "$BASE_OUT"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
BLOCK_LENGTH=32
REMASKING="low_confidence"
CFG="0.0"

# Prophet official/default hyperparameters. For this fair table we keep the
# user's task/prompt format fixed, so Prophet monitors an answer window but does
# not force the official gsm8k_cot_zeroshot "The answer is" phrase by default.
PROPHET_ANSWER_START="${PROPHET_ANSWER_START:-200}"
PROPHET_GSM_CONSTRAINTS="${PROPHET_GSM_CONSTRAINTS:-200:The|201:answer|202:is}"
PROPHET_USE_GSM_CONSTRAINTS="${PROPHET_USE_GSM_CONSTRAINTS:-false}"
PROPHET_ANSWER_LENGTH="${PROPHET_ANSWER_LENGTH:-5}"
PROPHET_EARLY_THRESHOLD="${PROPHET_EARLY_THRESHOLD:-7.5}"
PROPHET_MID_THRESHOLD="${PROPHET_MID_THRESHOLD:-5.0}"
PROPHET_LATE_THRESHOLD="${PROPHET_LATE_THRESHOLD:-2.5}"

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
  ! grep -Eq "Traceback|OutOfMemoryError|CUDA out of memory|returned non-zero exit status|Task.*not found|KeyError" "$log_file"
}

common_args () {
  local gen="$1"
  local steps="$2"
  echo "model_path=${MODEL_PATH},gen_length=${gen},steps=${steps},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG}"
}

prophet_args () {
  local task_family="$1"
  local gen="$2"
  local steps="$3"
  local common
  common="$(common_args "$gen" "$steps")"

  if [ "$task_family" = "gsm8k" ] && [ "$PROPHET_USE_GSM_CONSTRAINTS" = "true" ]; then
    echo "${common},method=Prophet,prophet_constraints_text=${PROPHET_GSM_CONSTRAINTS},prophet_answer_length=${PROPHET_ANSWER_LENGTH},prophet_early_threshold=${PROPHET_EARLY_THRESHOLD},prophet_mid_threshold=${PROPHET_MID_THRESHOLD},prophet_late_threshold=${PROPHET_LATE_THRESHOLD}"
  else
    echo "${common},method=Prophet,prophet_answer_start=${PROPHET_ANSWER_START},prophet_answer_length=${PROPHET_ANSWER_LENGTH},prophet_early_threshold=${PROPHET_EARLY_THRESHOLD},prophet_mid_threshold=${PROPHET_MID_THRESHOLD},prophet_late_threshold=${PROPHET_LATE_THRESHOLD}"
  fi
}

klass_thresholds () {
  local task_family="$1"
  case "$task_family" in
    gsm8k) echo "0.6 0.015" ;;
    math) echo "0.6 0.01" ;;
    humaneval) echo "0.9 0.01" ;;
    mbpp) echo "0.7 0.01" ;;
    *) echo "0.9 0.01" ;;
  esac
}

klass_args () {
  local task_family="$1"
  local gen="$2"
  local steps="$3"
  local conf kl common
  read -r conf kl < <(klass_thresholds "$task_family")
  common="$(common_args "$gen" "$steps")"
  echo "${common},method=KLASS,klass_alg=klass,klass_conf_threshold=${conf},klass_kl_threshold=${kl},klass_history_length=2,klass_unmask_strategy=all,klass_confidence_metric=prob"
}

dapd_best_hparams () {
  local task_family="$1"
  case "$task_family" in
    # Best DAPD paper mode per dataset, while keeping this script's block32
    # fair protocol fixed.
    gsm8k) echo "dapd_direct 0.005 0.05" ;;
    humaneval) echo "dapd_direct 0.01 0.05" ;;
    mbpp) echo "dapd_staged 0.01 0.15" ;;
    math) echo "dapd_staged 0.01 0.05" ;;
    *) echo "dapd_staged 0.01 0.15" ;;
  esac
}

dapd_args () {
  local task_family="$1"
  local gen="$2"
  local steps="$3"
  local alg tau_min tau_max common
  read -r alg tau_min tau_max < <(dapd_best_hparams "$task_family")
  common="$(common_args "$gen" "$steps")"
  echo "${common},method=DAPD,dapd_alg=${alg},dapd_layer_ratio=0.3,dapd_tau_min=${tau_min},dapd_tau_max=${tau_max}"
}

run_eval () {
  local setting="$1"
  local method_tag="$2"
  local task="$3"
  local fewshot="$4"
  local model_args="$5"

  local out_dir="${BASE_OUT}/${setting}"
  local log_file="${out_dir}/${method_tag}.log"
  mkdir -p "$out_dir"

  if is_complete "$log_file"; then
    echo ""
    echo "========== SKIP ${setting}/${method_tag}: completed =========="
    summarize_one "$log_file"
    return 0
  fi

  if [ -f "$log_file" ]; then
    mv "$log_file" "${log_file}.failed_$(date +%Y%m%d_%H%M%S)"
  fi

  echo ""
  echo "========== RUNNING ${setting}/${method_tag} =========="
  echo "TASK=${task}"
  echo "FEWSHOT=${fewshot}"
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

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e

  summarize_one "$log_file"
  return "$status"
}

run_method () {
  local method="$1"
  local setting="$2"
  local task="$3"
  local task_family="$4"
  local fewshot="$5"
  local gen="$6"
  local steps="$7"
  local conf kl alg tau_min tau_max

  case "$method" in
    prophet)
      run_eval "$setting" "prophet_besthp_fair_b32" \
        "$task" "$fewshot" "$(prophet_args "$task_family" "$gen" "$steps")"
      ;;
    klass)
      read -r conf kl < <(klass_thresholds "$task_family")
      run_eval "$setting" "klass_besthp_fair_b32_conf${conf/./}_kl${kl/./}" \
        "$task" "$fewshot" "$(klass_args "$task_family" "$gen" "$steps")"
      ;;
    dapd)
      read -r alg tau_min tau_max < <(dapd_best_hparams "$task_family")
      run_eval "$setting" "dapd_besthp_fair_b32_${alg}_tau${tau_min/./}_${tau_max/./}_headchunk${DAPD_ATTENTION_HEAD_CHUNK}" \
        "$task" "$fewshot" "$(dapd_args "$task_family" "$gen" "$steps")"
      ;;
    *)
      echo "Unknown method: $method" >&2
      return 1
      ;;
  esac
}

run_prophet_klass () {
  local setting="$1"
  local task="$2"
  local task_family="$3"
  local fewshot="$4"
  local gen="$5"
  local steps="$6"

  run_method prophet "$setting" "$task" "$task_family" "$fewshot" "$gen" "$steps"
  run_method klass "$setting" "$task" "$task_family" "$fewshot" "$gen" "$steps"
}

echo "========== Fair protocol =========="
echo "Model: ${MODEL_PATH}"
echo "Block length: ${BLOCK_LENGTH}"
echo "Output: ${BASE_OUT}"
echo "Phase 1: Prophet/KLASS first for every row"
echo "Phase 2: DAPD best-hparam fair runs for every row"

echo ""
echo "========== PHASE 1: Prophet/KLASS best hyperparams under fair block32 =========="
run_prophet_klass "gsm8k_g256_s256_full" "gsm8k" "gsm8k" "default" 256 256
run_prophet_klass "gsm8k_g1024_s1024_full" "gsm8k" "gsm8k" "default" 1024 1024
run_prophet_klass "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "math" "4" 256 256
run_prophet_klass "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "math" "4" 1024 1024
run_prophet_klass "humaneval_g256_s256_full" "humaneval" "humaneval" "default" 256 256
run_prophet_klass "humaneval_g1024_s1024_full" "humaneval" "humaneval" "default" 1024 1024
run_prophet_klass "mbpp_g256_s256_full" "mbpp" "mbpp" "default" 256 256
run_prophet_klass "mbpp_g1024_s1024_full" "mbpp" "mbpp" "default" 1024 1024

echo ""
echo "========== PHASE 2: DAPD best hyperparams under fair block32 =========="
run_method dapd "gsm8k_g256_s256_full" "gsm8k" "gsm8k" "default" 256 256
run_method dapd "gsm8k_g1024_s1024_full" "gsm8k" "gsm8k" "default" 1024 1024
run_method dapd "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "math" "4" 256 256
run_method dapd "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "math" "4" 1024 1024
run_method dapd "humaneval_g256_s256_full" "humaneval" "humaneval" "default" 256 256
run_method dapd "humaneval_g1024_s1024_full" "humaneval" "humaneval" "default" 1024 1024
run_method dapd "mbpp_g256_s256_full" "mbpp" "mbpp" "default" 256 256
run_method dapd "mbpp_g1024_s1024_full" "mbpp" "mbpp" "default" 1024 1024

echo ""
echo "========== FINAL SUMMARY: best hyperparams fair block32 =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|Prophet|KLASS|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
python extract_log_table.py "$BASE_OUT" || true
echo "Logs: ${BASE_OUT}"
