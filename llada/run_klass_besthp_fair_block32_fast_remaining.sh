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
export KLASS_PROB_DTYPE="${KLASS_PROB_DTYPE:-float32}"
export KLASS_KL_VOCAB_CHUNK="${KLASS_KL_VOCAB_CHUNK:-8192}"

BASE_OUT="output/ablations/klass_besthp_fair_block32_full"
OLD_GSM8K_256="output/ablations/best_hparams_fair_block32_full/gsm8k_g256_s256_full/klass_besthp_fair_b32_conf06_kl0015.log"
mkdir -p "$BASE_OUT"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
BLOCK_LENGTH=32
REMASKING="low_confidence"
CFG="0.0"

summarize_one () {
  local log_file="$1"
  echo ""
  echo "========== SUMMARY: ${log_file} =========="
  grep -E "Number of tokens|Generation time|Tokens per second|KLASS actual denoising step ratio|KLASS ready tokens|KLASS fallback tokens|KLASS avg KL|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$log_file" || true
  echo "========== END SUMMARY =========="
}

is_complete () {
  local log_file="$1"
  [ -f "$log_file" ] || return 1
  grep -Eq "Number of tokens|Generation time|Tokens per second" "$log_file" || return 1
  grep -Eq "exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$log_file" || return 1
  ! grep -Eq "Traceback|OutOfMemoryError|CUDA out of memory|returned non-zero exit status|Task.*not found|KeyError" "$log_file"
}

make_klass_args () {
  local gen="$1"
  local steps="$2"
  local conf="$3"
  local kl="$4"
  echo "model_path=${MODEL_PATH},gen_length=${gen},steps=${steps},block_length=${BLOCK_LENGTH},remasking=${REMASKING},cfg=${CFG},method=KLASS,klass_alg=klass,klass_conf_threshold=${conf},klass_kl_threshold=${kl},klass_history_length=2,klass_unmask_strategy=all,klass_confidence_metric=prob"
}

run_eval () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local gen="$4"
  local steps="$5"
  local conf="$6"
  local kl="$7"

  local out_dir="${BASE_OUT}/${setting}"
  local log_file="${out_dir}/klass_conf${conf/./}_kl${kl/./}.log"
  mkdir -p "$out_dir"

  if is_complete "$log_file"; then
    echo ""
    echo "========== SKIP ${setting}: completed =========="
    summarize_one "$log_file"
    return 0
  fi

  if [ -f "$log_file" ]; then
    mv "$log_file" "${log_file}.incomplete_$(date +%Y%m%d_%H%M%S)"
  fi

  local model_args
  model_args="$(make_klass_args "$gen" "$steps" "$conf" "$kl")"

  echo ""
  echo "========== RUNNING ${setting} =========="
  echo "TASK=${task}"
  echo "FEWSHOT=${fewshot}"
  echo "GEN=${gen}"
  echo "STEPS=${steps}"
  echo "BLOCK_LENGTH=${BLOCK_LENGTH}"
  echo "CONF=${conf}"
  echo "KL=${kl}"
  echo "KLASS_PROB_DTYPE=${KLASS_PROB_DTYPE}"
  echo "KLASS_KL_VOCAB_CHUNK=${KLASS_KL_VOCAB_CHUNK}"
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

echo "========== Existing completed rows =========="
for log_file in \
  "$OLD_GSM8K_256" \
  "${BASE_OUT}/gsm8k_g1024_s1024_full/klass_conf06_kl0015.log" \
  "${BASE_OUT}/math_g256_s256_4shot_full/klass_conf06_kl001.log"; do
  if is_complete "$log_file"; then
    summarize_one "$log_file"
  else
    echo "Not complete or missing: ${log_file}"
  fi
done

echo ""
echo "========== Fast remaining rows first =========="
run_eval "humaneval_g256_s256_full" "humaneval" "default" 256 256 "0.9" "0.01"
run_eval "humaneval_g1024_s1024_full" "humaneval" "default" 1024 1024 "0.9" "0.01"
run_eval "mbpp_g256_s256_full" "mbpp" "default" 256 256 "0.7" "0.01"
run_eval "mbpp_g1024_s1024_full" "mbpp" "default" 1024 1024 "0.7" "0.01"

echo ""
echo "========== Slow row last =========="
run_eval "math_g1024_s1024_4shot_full" "score_non_greedy_robustness_math" "4" 1024 1024 "0.6" "0.01"

echo ""
echo "========== FINAL SUMMARY: KLASS best hparams fair block32 =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|KLASS actual denoising step ratio|KLASS ready tokens|KLASS fallback tokens|KLASS avg KL|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log "$OLD_GSM8K_256" 2>/dev/null || true
python extract_log_table.py "$BASE_OUT" || true
echo "Logs: ${BASE_OUT}"
