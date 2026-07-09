#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
TARGET_FFN_RATIO="${TARGET_FFN_RATIO:-0.70}"
GEN_LENGTH="${GEN_LENGTH:-128}"
STEPS="${STEPS:-$GEN_LENGTH}"
BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
BUDGET_SCHEDULE="${BUDGET_SCHEDULE:-cosine}"
ITERATIONS="${ITERATIONS:-5}"

python calibrate_cora_budget.py \
  --model_path "$MODEL_PATH" \
  --target_ffn_ratio "$TARGET_FFN_RATIO" \
  --gen_length "$GEN_LENGTH" \
  --steps "$STEPS" \
  --block_length "$BLOCK_LENGTH" \
  --budget_schedule "$BUDGET_SCHEDULE" \
  --iterations "$ITERATIONS"
