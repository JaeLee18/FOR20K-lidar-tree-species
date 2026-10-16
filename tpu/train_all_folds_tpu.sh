#!/usr/bin/env bash
# Train all five folds of one configuration on a Cloud TPU, then evaluate and export the weights.
#   bash tpu/train_all_folds_tpu.sh configs/convnextv2_256.yaml runs/tpu_convnextv2_256 [FIRST_FOLD] [LAST_FOLD]
#   TRAIN_ARGS="--per-chip-batch 4" bash tpu/train_all_folds_tpu.sh configs/convnextv2_512.yaml runs/tpu_convnextv2_512
# Re-running resumes unfinished folds and skips finished ones. The weights are exported to
# weights_tpu/<config name>/, next to the released ones rather than over them; predict with
# them through --weights weights_tpu.
set -euo pipefail
CONFIG=${1:?config yaml}
OUT=${2:?output dir}
FIRST=${3:-0}
LAST=${4:-4}
PY=${PY:-python}
WEIGHTS_OUT=${WEIGHTS_OUT:-weights_tpu}
cd "$(dirname "$0")/.."
mkdir -p "$OUT"

for FOLD in $(seq "$FIRST" "$LAST"); do
  if [[ -f "$OUT/fold${FOLD}.done" ]]; then echo "fold $FOLD already done"; continue; fi
  echo "[$(date '+%F %T')] fold $FOLD"
  # shellcheck disable=SC2086  # TRAIN_ARGS holds several arguments
  "$PY" tpu/train_tpu.py --config "$CONFIG" --fold "$FOLD" --out "$OUT" --resume ${TRAIN_ARGS:-}
  touch "$OUT/fold${FOLD}.done"
done

"$PY" scripts/eval/evaluate_oof.py "$OUT"
SIZE=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['img_size'])" "$CONFIG")
NAME=$(basename "$CONFIG" .yaml)
"$PY" scripts/export/export_weights.py --ckpt-dir "$OUT" --img-size "$SIZE" --out "$WEIGHTS_OUT/$NAME"
echo "exported $WEIGHTS_OUT/$NAME"
