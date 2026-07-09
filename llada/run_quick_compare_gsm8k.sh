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
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs_quick

TASK=gsm8k
LIMIT=20
GEN=256
STEPS=256
BLOCK=32

echo "===== Running Original LLaDA ====="
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=30001 \
accelerate launch eval_llada.py \
  --tasks ${TASK} \
  --model llada_dist \
  --confirm_run_unsafe_code \
  --model_args "model_path=GSAI-ML/LLaDA-8B-Instruct,gen_length=${GEN},steps=${STEPS},block_length=${BLOCK},method=original" \
  --limit ${LIMIT} \
  2>&1 | tee logs_quick/original_${TASK}_limit${LIMIT}_g${GEN}_s${STEPS}.log

echo "===== Running CORA-Diff ====="
RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=30002 \
TASK=${TASK} LIMIT=${LIMIT} GEN_LENGTH=${GEN} STEPS=${STEPS} BLOCK_LENGTH=${BLOCK} \
CORA_ACTIVE_RATIO=0.7 CORA_ACTIVE_RATIO_FINAL=0.45 \
bash run_cora_gsm8k.sh \
  2>&1 | tee logs_quick/cora_${TASK}_limit${LIMIT}_g${GEN}_s${STEPS}.log

echo "===== Extract key results ====="
grep -E "Tokens per second|Generation time|CORA avg refinement ratio|CORA estimated activated FFN ratio|exact_match" logs_quick/*.log

echo "Done."
