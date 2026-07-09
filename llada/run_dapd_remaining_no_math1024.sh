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

# Algorithm-equivalent memory guard: DAPD computes attention heads in chunks
# instead of one large QK tensor. Smaller chunks are slower but safer for MBPP 1024.
export DAPD_ATTENTION_HEAD_CHUNK="${DAPD_ATTENTION_HEAD_CHUNK:-2}"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"

BASE_OUT="output/ablations/all_tasks_dapd_fair_block32_full_resume_headchunk${DAPD_ATTENTION_HEAD_CHUNK}"
mkdir -p "$BASE_OUT"

make_dapd_args () {
  local gen="$1"
  local steps="$2"
  local tau_min="$3"
  local tau_max="$4"

  echo "model_path=${MODEL_PATH},gen_length=${gen},steps=${steps},block_length=32,remasking=low_confidence,cfg=0.0,method=DAPD,dapd_alg=dapd_staged,dapd_layer_ratio=0.3,dapd_tau_min=${tau_min},dapd_tau_max=${tau_max}"
}

summarize_one () {
  local log_file="$1"

  echo ""
  echo "========== SUMMARY: ${log_file} =========="
  grep -E "Number of tokens|Generation time|Tokens per second|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$log_file" || true
  echo "========== END SUMMARY =========="
}

is_complete () {
  local log_file="$1"
  [ -f "$log_file" ] || return 1
  grep -Eq "Number of tokens|Generation time|Tokens per second" "$log_file" || return 1
  grep -Eq "exact_match|pass_at_1|pass@1|non_greedy_accuracy" "$log_file" || return 1
  ! grep -Eq "Traceback|OutOfMemoryError|CUDA out of memory|returned non-zero exit status|Killed" "$log_file"
}

run_eval () {
  local setting="$1"
  local task="$2"
  local fewshot="$3"
  local gen="$4"
  local steps="$5"
  local tau_min="$6"
  local tau_max="$7"
  local out_dir="${BASE_OUT}/${setting}"

  local tau_min_tag="${tau_min/./}"
  local tau_max_tag="${tau_max/./}"
  local name="dapd_staged_tau${tau_min_tag}_${tau_max_tag}_full_headchunk${DAPD_ATTENTION_HEAD_CHUNK}"
  local log_file="${out_dir}/${name}.log"
  mkdir -p "$out_dir"

  if is_complete "$log_file"; then
    echo ""
    echo "========== SKIP ${setting}/${name}: completed =========="
    summarize_one "$log_file"
    return 0
  fi

  if [ -f "$log_file" ]; then
    mv "$log_file" "${log_file}.incomplete_$(date +%Y%m%d_%H%M%S)"
  fi

  local model_args
  model_args="$(make_dapd_args "$gen" "$steps" "$tau_min" "$tau_max")"

  echo ""
  echo "========== RUNNING ${setting}/${name} =========="
  echo "TASK=${task}"
  echo "FEWSHOT=${fewshot}"
  echo "GEN=${gen}"
  echo "STEPS=${steps}"
  echo "BLOCK_LENGTH=32"
  echo "DAPD_ATTENTION_HEAD_CHUNK=${DAPD_ATTENTION_HEAD_CHUNK}"
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

echo "========== DAPD remaining rows, excluding MATH 1024/1024 =========="

run_eval "mbpp_g1024_s1024_full" "mbpp" "default" 1024 1024 "0.01" "0.15"

run_eval "math_g256_s256_4shot_full" "score_non_greedy_robustness_math" "4" 256 256 "0.01" "0.05"

echo ""
echo "========== FINAL SUMMARY: DAPD remaining no MATH1024 headchunk${DAPD_ATTENTION_HEAD_CHUNK} =========="
grep -R -E "Number of tokens|Generation time|Tokens per second|DAPD avg total steps|DAPD step ratio vs configured steps|DAPD avg tokens per step|flexible-extract|strict-match|exact_match|pass_at_1|pass@1|non_greedy_accuracy" "${BASE_OUT}"/*/*.log || true
echo "========== END FINAL SUMMARY =========="
