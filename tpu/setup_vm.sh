#!/usr/bin/env bash
# Install the TPU environment on a Cloud TPU VM and check that the chips are visible.
#
#   bash tpu/setup_vm.sh                   # creates .venv-tpu in the repository
#   VENV=/opt/venv bash tpu/setup_vm.sh    # somewhere else
#
# Then: source .venv-tpu/bin/activate && python tpu/predict_tpu.py examples/
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$REPO/.venv-tpu}"

if ! python3 -m venv "$VENV"; then
  echo "[setup] python3 -m venv failed; install it with: sudo apt-get install -y python3-venv" >&2
  exit 1
fi
PIP=("$VENV/bin/pip" install --quiet --disable-pip-version-check)
"${PIP[@]}" --upgrade pip
"${PIP[@]}" torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
"${PIP[@]}" -r "$REPO/tpu/requirements.txt"
"${PIP[@]}" --no-deps -e "$REPO"

"$VENV/bin/python" - <<'EOF'
import torch, torch_xla, torch_xla.runtime as xr
print(f"torch {torch.__version__} | torch_xla {torch_xla.__version__} | "
      f"{xr.device_type()} x {xr.global_runtime_device_count()} devices")
EOF
echo "[setup] done: source $VENV/bin/activate"
