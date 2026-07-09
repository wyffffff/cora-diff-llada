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
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DAPD_ATTENTION_HEAD_CHUNK="${DAPD_ATTENTION_HEAD_CHUNK:-4}"

BASE_PK="${BASE_PK:-output/ablations/prophet_klass_checked_fair_block32_full}"
BASE_DAPD="${BASE_DAPD:-output/ablations/all_tasks_dapd_fair_block32_full_resume_headchunk${DAPD_ATTENTION_HEAD_CHUNK}}"
mkdir -p "$BASE_PK" "$BASE_DAPD"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
FAIR_BLOCK_LENGTH="${FAIR_BLOCK_LENGTH:-32}"
KLASS_BLOCK_LENGTH="${KLASS_BLOCK_LENGTH:-$FAIR_BLOCK_LENGTH}"
REMASKING="low_confidence"
CFG="0.0"

PROPHET_ANSWER_START="${PROPHET_ANSWER_START:-200}"
PROPHET_GSM_CONSTRAINTS="${PROPHET_GSM_CONSTRAINTS:-200:The|201:answer|202:is}"
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
  local block="$3"
  echo "model_path=${MODEL_PATH},gen_length=${gen},steps=${steps},block_length=${block},remasking=${REMASKING},cfg=${CFG}"
}

prophet_args () {
  local task_family="$1"
  local gen="$2"
  local steps="$3"
  local common
  common="$(common_args "$gen" "$steps" "$FAIR_BLOCK_LENGTH")"

  if [ "$task_family" = "gsm8k" ]; then
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
  common="$(common_args "$gen" "$steps" "$KLASS_BLOCK_LENGTH")"
  echo "${common},method=KLASS,klass_alg=klass,klass_conf_threshold=${conf},klass_kl_threshold=${kl},klass_history_length=2,klass_unmask_strategy=all,klass_confidence_metric=prob"
}

dapd_args () {
  local gen="$1"
  local steps="$2"
  local tau_min="$3"
  local tau_max="$4"
  local common
  common="$(common_args "$gen" "$steps" "$FAIR_BLOCK_LENGTH")"
  echo "${common},method=DAPD,dapd_alg=dapd_staged,dapd_layer_ratio=0.3,dapd_tau_min=${tau_min},dapd_tau_max=${tau_max}"
}

run_eval () {
  local out_root="$1"
  local setting="$2"
  local log_name="$3"
  local task="$4"
  local fewshot="$5"
  local model_args="$6"

  local out_dir="${out_root}/${setting}"
  local log_file="${out_dir}/${log_name}.log"
  mkdir -p "$out_dir"

  if is_complete "$log_file"; then
    echo ""
    echo "========== SKIP ${setting}/${log_name}: completed =========="
    summarize_one "$log_file"
    return 0
  fi

  if [ -f "$log_file" ]; then
    mv "$log_file" "${log_file}.failed_$(date +%Y%m%d_%H%M%S)"
  fi

  echo ""
  echo "========== RUNNING ${setting}/${log_name} =========="
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

run_prophet_klass () {
  local setting="$1"
  local task="$2"
  local task_family="$3"
  local fewshot="$4"
  local gen="$5"
  local steps="$6"
  local conf kl
  read -r conf kl < <(klass_thresholds "$task_family")

  run_eval "$BASE_PK" "$setting" "prophet_checked_b${FAIR_BLOCK_LENGTH}" \
    "$task" "$fewshot" \
    "$(prophet_args "$task_family" "$gen" "$steps")"

  run_eval "$BASE_PK" "$setting" "klass_checked_b${KLASS_BLOCK_LENGTH}_conf${conf/./}_kl${kl/./}" \
    "$task" "$fewshot" \
    "$(klass_args "$task_family" "$gen" "$steps")"
}

run_dapd_remaining () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local gen="$4"
  local steps="$5"
  local tau_min="$6"
  local tau_max="$7"
  local tau_min_tag="${tau_min/./}"
  local tau_max_tag="${tau_max/./}"

  run_eval "$BASE_DAPD" "$setting" "dapd_staged_tau${tau_min_tag}_${tau_max_tag}_full_headchunk${DAPD_ATTENTION_HEAD_CHUNK}" \
    "$task" "$fewshot" \
    "$(dapd_args "$gen" "$steps" "$tau_min" "$tau_max")"
}

run_all_three_remaining () {
  local setting="$1"
  local task="$2"
  local task_family="$3"
  local fewshot="$4"
  local gen="$5"
  local steps="$6"
  local tau_min="$7"
  local tau_max="$8"

  run_prophet_klass "$setting" "$task" "$task_family" "$fewshot" "$gen" "$steps"
  run_dapd_remaining "$setting" "$task" "$fewshot" "$gen" "$steps" "$tau_min" "$tau_max"
}

echo "========== PHASE 1: Prophet/KLASS checked configs for rows where DAPD already finished =========="
run_prophet_klass "gsm8k_g256_s256_full" "gsm8k" "gsm8k" "default" 256 256
run_prophet_klass "gsm8k_g1024_s1024_full" "gsm8k" "gsm8k" "default" 1024 1024
run_prophet_klass "humaneval_g256_s256_full" "humaneval" "humaneval" "default" 256 256
run_prophet_klass "humaneval_g1024_s1024_full" "humaneval" "humaneval" "default" 1024 1024
run_prophet_klass "mbpp_g256_s256_full" "mbpp" "mbpp" "default" 256 256

echo ""
echo "========== PHASE 2: remaining rows, run Prophet/KLASS/DAPD =========="
run_all_three_remaining "mbpp_g1024_s1024_full" "mbpp" "mbpp" "default" 1024 1024 "0.01" "0.15"
run_all_three_remaining "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "math" "4" 256 256 "0.01" "0.05"
run_all_three_remaining "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "math" "4" 1024 1024 "0.01" "0.05"

echo ""
echo "========== FINAL SUMMARY: Prophet/KLASS checked + remaining DAPD =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|Prophet|KLASS|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$BASE_PK"/*/*.log "$BASE_DAPD"/*/*.log 2>/dev/null || true
python extract_log_table.py "$BASE_PK" || true
python extract_log_table.py "$BASE_DAPD" || true
echo "Prophet/KLASS logs: ${BASE_PK}"
echo "Remaining DAPD logs: ${BASE_DAPD}"
