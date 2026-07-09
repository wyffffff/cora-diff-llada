#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TASK="${TASK:-gsm8k}"
export LIMIT="${LIMIT:-20}"
export GEN_LENGTH="${GEN_LENGTH:-256}"
export STEPS="${STEPS:-$GEN_LENGTH}"
export BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
export CORA_ROUTING_MODE=disabled
export CORA_ORDER_CHANNELS=false
export CORA_EXTRA_COMMIT=false
export CORA_ALPHA=0.0
export CORA_DEPENDENCY_TOPK=0
export CORA_FAST_ACCEPT=true
export CORA_ACCEPT_CONFIDENCE_THRESHOLD="${CORA_ACCEPT_CONFIDENCE_THRESHOLD:-0.97}"
export CORA_ACCEPT_STABILITY_STEPS="${CORA_ACCEPT_STABILITY_STEPS:-2}"
export CORA_EOT=false

OUT_DIR="output/residual_sweep/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

run_case() {
  local name="$1"
  local residual_accept="$2"
  local residual_threshold="$3"
  local beta="$4"
  local gamma="$5"
  local topk="$6"
  local log_file="${OUT_DIR}/${name}.log"

  echo "========== ${name} =========="
  echo "residual_accept=${residual_accept} threshold=${residual_threshold} beta=${beta} gamma=${gamma} topk=${topk}"

  CORA_RESIDUAL_ACCEPT="$residual_accept" \
  CORA_RESIDUAL_THRESHOLD="$residual_threshold" \
  CORA_RESIDUAL_BETA="$beta" \
  CORA_RESIDUAL_GAMMA="$gamma" \
  CORA_DRIFT_TOPK="$topk" \
  bash run_cora_gsm8k.sh 2>&1 | tee "$log_file"
}

# Current validated path: confidence + prediction persistence only.
run_case "persistence_only" "false" "0.0" "0.0" "0.0" "1"

# Residual-score variants. Thresholds are intentionally swept because r_i is
# z-score based and task/model dependent.
run_case "residual_t0_top8" "true" "0.0" "1.0" "1.0" "8"
run_case "residual_t05_top8" "true" "0.5" "1.0" "1.0" "8"
run_case "residual_t10_top8" "true" "1.0" "1.0" "1.0" "8"
run_case "drift_only_t05_top8" "true" "0.5" "1.0" "0.0" "8"

echo "========== summary =========="
grep -E "Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA residual accept tokens|CORA avg residual score|exact_match" "${OUT_DIR}"/*.log || true
echo "Logs: ${OUT_DIR}"
