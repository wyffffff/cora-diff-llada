#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${ENV_NAME:-cora}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"

export HF_HOME="${HF_HOME:-$SCRIPT_DIR/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
  fi
  conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip

if [[ "$INSTALL_TORCH" == "1" ]]; then
  python -m pip install -r requirements.txt
else
  python -m pip install -r requirements-autodl.txt
fi

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "Setup complete."
echo "Before running later: conda activate ${ENV_NAME}"
