#!/usr/bin/env bash
# Classify the species of single-tree LiDAR scans.
#
#   ./predict.sh examples/05554_Pinus_sylvestris_ULS.laz
#   ./predict.sh my_scans/ -o predictions.csv
#   ./predict.sh tree.ply --resolution 256 --tta 1 --device cpu
#   ./predict.sh --docker tree.laz          # run inside the Docker image
#
# Every option after the inputs is passed to `mvtree predict` (see --help).
# Accepted formats: .las .laz .ply .pcd .xyz .txt .csv .pts .npy .npz
# One file = one segmented tree, coordinates in metres with Z up.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${MVTREE_IMAGE:-three-lidars:latest}"

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  echo
  echo "Options (forwarded to mvtree predict):"
  echo "  -o, --output FILE        write results to .csv or .json"
  echo "  -k, --topk K             number of species to report (default 5)"
  echo "  -r, --resolution R       auto | 256 | 512 | multi (default auto = all installed)"
  echo "      --tta N              augmented passes per model (default 20, 1 = off)"
  echo "      --folds 0,1          subset of the five fold models"
  echo "      --device D           auto | cuda | cpu (default auto)"
  echo "      --low-memory         keep one network in memory at a time"
  echo "      --save-renders DIR   also write the 7 rendered views + a preview"
  echo
  echo "Environment: MVTREE_WEIGHTS (weights dir), MVTREE_GDRIVE_URL (weights link), MVTREE_IMAGE (Docker image)"
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi

USE_DOCKER=0
ARGS=()
for a in "$@"; do
  if [[ "$a" == "--docker" ]]; then USE_DOCKER=1; else ARGS+=("$a"); fi
done

# ---------------------------------------------------------------- docker mode
if [[ "$USE_DOCKER" == 1 ]]; then
  command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
  MOUNTS=(-v "$REPO/weights:/app/weights")
  DOCKER_ARGS=()
  for a in "${ARGS[@]}"; do
    if [[ -e "$a" ]]; then
      abs="$(cd "$(dirname "$a")" && pwd)/$(basename "$a")"
      MOUNTS+=(-v "$(dirname "$abs"):$(dirname "$abs")")
      DOCKER_ARGS+=("$abs")
    else
      DOCKER_ARGS+=("$a")
    fi
  done
  GPU=()
  if command -v nvidia-smi >/dev/null 2>&1; then GPU=(--gpus all); fi
  exec docker run --rm "${GPU[@]}" --shm-size 2g -w "$PWD" -v "$PWD:$PWD" "${MOUNTS[@]}" \
    -e MVTREE_GDRIVE_URL="${MVTREE_GDRIVE_URL:-}" "$IMAGE" predict "${DOCKER_ARGS[@]}"
fi

# ----------------------------------------------------------------- local mode
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY=("$REPO/.venv/bin/python")
elif command -v uv >/dev/null 2>&1 && [[ -f "$REPO/pyproject.toml" ]]; then
  PY=(uv run --project "$REPO" python)
else
  PY=(python3)
fi
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

if ! "${PY[@]}" -c "import mvtree, torch, timm, laspy" >/dev/null 2>&1; then
  echo "[predict.sh] dependencies missing. Install with one of:" >&2
  echo "    uv sync                      # recommended" >&2
  echo "    pip install -e .             # or: conda env create -f requirements/environment.yml" >&2
  echo "    ./predict.sh --docker ...    # or run in Docker" >&2
  exit 1
fi

WEIGHTS_DIR="${MVTREE_WEIGHTS:-$REPO/weights}"
if ! ls "$WEIGHTS_DIR"/convnextv2_*/fold*.safetensors >/dev/null 2>&1; then
  echo "[predict.sh] no weights in $WEIGHTS_DIR, downloading from Google Drive..." >&2
  "${PY[@]}" -m mvtree download --weights "$WEIGHTS_DIR" || {
    echo "[predict.sh] download failed. Fetch the weights manually (weights/README.md)" >&2
    echo "             and place them under $WEIGHTS_DIR/convnextv2_{256,512}/" >&2
    exit 1
  }
fi

exec "${PY[@]}" -m mvtree predict "${ARGS[@]}" --weights "$WEIGHTS_DIR"
