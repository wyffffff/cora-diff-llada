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

BASE_OUT="${BASE_OUT:-output/ablations/official_best_configs_256_full}"
mkdir -p "$BASE_OUT"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
GEN_LENGTH=256
STEPS=256
REMASKING="low_confidence"
CFG="0.0"

# Prophet's public LLaDA eval script only gives a GSM8K setting. The other
# tasks are left out here instead of inventing a fake "official best" config.
PROPHET_GSM_TASK="${PROPHET_GSM_TASK:-gsm8k}"
PROPHET_CONSTRAINTS="${PROPHET_CONSTRAINTS:-200:The|201:answer|202:is}"
PROPHET_ANSWER_LENGTH="${PROPHET_ANSWER_LENGTH:-5}"
PROPHET_EARLY_THRESHOLD="${PROPHET_EARLY_THRESHOLD:-7.5}"
PROPHET_MID_THRESHOLD="${PROPHET_MID_THRESHOLD:-5.0}"
PROPHET_LATE_THRESHOLD="${PROPHET_LATE_THRESHOLD:-2.5}"

# KLASS official scripts use dataset-specific thresholds and block_length=64.
KLASS_GSM_TASK="${KLASS_GSM_TASK:-gsm8k}"
KLASS_MATH_TASK="${KLASS_MATH_TASK:-score_non_greedy_robustness_math}"
KLASS_MATH_FEWSHOT="${KLASS_MATH_FEWSHOT:-4}"
KLASS_HUMANEVAL_TASK="${KLASS_HUMANEVAL_TASK:-humaneval}"
KLASS_MBPP_TASK="${KLASS_MBPP_TASK:-mbpp}"

# DAPD paper uses Math500. If this lm-eval environment does not provide it,
# set DAPD_MATH_TASK=score_non_greedy_robustness_math before running.
DAPD_GSM_TASK="${DAPD_GSM_TASK:-gsm8k}"
DAPD_MATH_TASK="${DAPD_MATH_TASK:-minerva_math500}"
DAPD_MATH_FEWSHOT="${DAPD_MATH_FEWSHOT:-default}"
DAPD_HUMANEVAL_TASK="${DAPD_HUMANEVAL_TASK:-humaneval}"
DAPD_MBPP_TASK="${DAPD_MBPP_TASK:-mbpp}"

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
  local block="$1"
  echo "model_path=${MODEL_PATH},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${block},remasking=${REMASKING},cfg=${CFG}"
}

run_eval () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local model_args="$4"

  local out_dir="${BASE_OUT}/${setting}"
  local log_file="${out_dir}/run.log"
  mkdir -p "$out_dir"

  if is_complete "$log_file"; then
    echo ""
    echo "========== SKIP ${setting}: completed =========="
    summarize_one "$log_file"
    return 0
  fi

  if [ -f "$log_file" ]; then
    mv "$log_file" "${log_file}.failed_$(date +%Y%m%d_%H%M%S)"
  fi

  echo ""
  echo "========== RUNNING ${setting} =========="
  echo "TASK=${task}"
  echo "FEWSHOT=${fewshot}"
  echo "GEN=${GEN_LENGTH}"
  echo "STEPS=${STEPS}"
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

run_eval_allow_task_fallback () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local fallback_task="$4"
  local fallback_fewshot="$5"
  local model_args="$6"

  set +e
  run_eval "$setting" "$task" "$fewshot" "$model_args"
  local status=$?
  set -e
  if [ "$status" -eq 0 ]; then
    return 0
  fi

  echo ""
  echo "========== ${setting} failed with task=${task}; trying fallback task=${fallback_task} =========="
  run_eval "${setting}_fallback_${fallback_task}" "$fallback_task" "$fallback_fewshot" "$model_args"
}

prophet_args () {
  local common
  common="$(common_args 32)"
  echo "${common},method=Prophet,prophet_constraints_text=${PROPHET_CONSTRAINTS},prophet_answer_length=${PROPHET_ANSWER_LENGTH},prophet_early_threshold=${PROPHET_EARLY_THRESHOLD},prophet_mid_threshold=${PROPHET_MID_THRESHOLD},prophet_late_threshold=${PROPHET_LATE_THRESHOLD}"
}

klass_args () {
  local conf="$1"
  local kl="$2"
  local common
  common="$(common_args 64)"
  echo "${common},method=KLASS,klass_alg=klass,klass_conf_threshold=${conf},klass_kl_threshold=${kl},klass_history_length=2,klass_unmask_strategy=all,klass_confidence_metric=prob"
}

dapd_args () {
  local alg="$1"
  local block="$2"
  local tau_min="$3"
  local tau_max="$4"
  local common
  common="$(common_args "$block")"
  echo "${common},method=DAPD,dapd_alg=${alg},dapd_layer_ratio=0.3,dapd_tau_min=${tau_min},dapd_tau_max=${tau_max}"
}

echo "========== Prophet official config =========="
run_eval "prophet_gsm8k_official_g256_b32" \
  "$PROPHET_GSM_TASK" "default" \
  "$(prophet_args)"

echo ""
echo "========== KLASS official per-task configs =========="
run_eval "klass_gsm8k_official_g256_b64_conf060_kl0015" \
  "$KLASS_GSM_TASK" "default" \
  "$(klass_args 0.6 0.015)"

run_eval "klass_math_official_g256_b64_conf060_kl001" \
  "$KLASS_MATH_TASK" "$KLASS_MATH_FEWSHOT" \
  "$(klass_args 0.6 0.01)"

run_eval "klass_humaneval_official_g256_b64_conf090_kl001" \
  "$KLASS_HUMANEVAL_TASK" "default" \
  "$(klass_args 0.9 0.01)"

run_eval "klass_mbpp_official_g256_b64_conf070_kl001" \
  "$KLASS_MBPP_TASK" "default" \
  "$(klass_args 0.7 0.01)"

echo ""
echo "========== DAPD paper best-by-accuracy configs =========="
run_eval "dapd_gsm8k_paper_best_direct_4block_g256_b64_tau0005_005" \
  "$DAPD_GSM_TASK" "default" \
  "$(dapd_args dapd_direct 64 0.005 0.05)"

run_eval "dapd_humaneval_paper_best_direct_4block_g256_b64_tau001_005" \
  "$DAPD_HUMANEVAL_TASK" "0" \
  "$(dapd_args dapd_direct 64 0.01 0.05)"

run_eval "dapd_mbpp_paper_best_staged_1block_g256_b256_tau001_015" \
  "$DAPD_MBPP_TASK" "default" \
  "$(dapd_args dapd_staged 256 0.01 0.15)"

run_eval_allow_task_fallback "dapd_math500_paper_best_staged_1block_g256_b256_tau001_005" \
  "$DAPD_MATH_TASK" "$DAPD_MATH_FEWSHOT" \
  "score_non_greedy_robustness_math" "4" \
  "$(dapd_args dapd_staged 256 0.01 0.05)"

echo ""
echo "========== FINAL SUMMARY: official best configs =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|Prophet|KLASS|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/run.log || true
python extract_log_table.py "$BASE_OUT" || true
echo "Logs: ${BASE_OUT}"
