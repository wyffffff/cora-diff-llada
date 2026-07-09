#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========== Main-table reproduction =========="
echo "This script runs the full fair block32 Prophet/KLASS/DAPD table."
echo "It can take many GPU-hours on LLaDA-8B. For a quick sanity check, run:"
echo "  bash run_smoke_open_source.sh"
echo ""

exec bash run_best_hparams_fair_block32_full.sh
