#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export TASK="${TASK:-gsm8k}"
export LIMIT="${LIMIT:-20}"
export GEN_LENGTH="${GEN_LENGTH:-128}"
export STEPS="${STEPS:-$GEN_LENGTH}"
export BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
export CORA_ROUTING_MODE=disabled
export CORA_ORDER_CHANNELS=false
export CORA_EXTRA_COMMIT=false
export CORA_ALPHA=0.0
export CORA_DEPENDENCY_TOPK=0
export CORA_FAST_ACCEPT=true

OUT_DIR="output/step_sweep/${TASK}_g${GEN_LENGTH}_s${STEPS}_limit${LIMIT}"
mkdir -p "$OUT_DIR"

run_case() {
  local name="$1"
  local accept_threshold="$2"
  local stability_steps="$3"
  local eot_enabled="$4"
  local eot_threshold="$5"
  local log_file="${OUT_DIR}/${name}.log"

  echo "========== ${name} =========="
  echo "accept=${accept_threshold} stability=${stability_steps} eot=${eot_enabled} eot_threshold=${eot_threshold}"

  CORA_ACCEPT_CONFIDENCE_THRESHOLD="$accept_threshold" \
  CORA_ACCEPT_STABILITY_STEPS="$stability_steps" \
  CORA_EOT="$eot_enabled" \
  CORA_EOT_CONFIDENCE_THRESHOLD="$eot_threshold" \
  bash run_cora_gsm8k.sh 2>&1 | tee "$log_file"
}

run_case "accept090_eot050" "0.90" "1" "true" "0.50"
run_case "accept095_eot070" "0.95" "1" "true" "0.70"
run_case "accept097_stab2_eot080" "0.97" "2" "true" "0.80"
run_case "accept097_stab2_noeot" "0.97" "2" "false" "0.80"

echo "Sweep complete. Logs: ${OUT_DIR}"
grep -E "Running CORA-Diff|LIMIT=|Tokens per second|CORA actual denoising step ratio|CORA fast accept tokens|CORA EoT truncated samples|exact_match" "${OUT_DIR}"/*.log || true
