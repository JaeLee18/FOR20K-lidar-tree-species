#!/usr/bin/env bash
# Train all five folds of one configuration, then export the release weights.
#   bash scripts/train/train_all_folds.sh configs/convnextv2_512.yaml runs/convnextv2_512 [FIRST_FOLD] [LAST_FOLD]
# Re-running resumes unfinished folds and skips finished ones.
set -euo pipefail
CONFIG=${1:?config yaml}
OUT=${2:?output dir}
FIRST=${3:-0}
LAST=${4:-4}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT"

for FOLD in $(seq "$FIRST" "$LAST"); do
  if [[ -f "$OUT/fold${FOLD}.done" ]]; then echo "fold $FOLD already done"; continue; fi
  echo "[$(date '+%F %T')] fold $FOLD"
  python scripts/train/train.py --config "$CONFIG" --fold "$FOLD" --out "$OUT" --resume
  touch "$OUT/fold${FOLD}.done"
done

python scripts/eval/evaluate_oof.py "$OUT"
SIZE=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['img_size'])" "$CONFIG")
NAME=$(basename "$CONFIG" .yaml)
python scripts/export/export_weights.py --ckpt-dir "$OUT" --img-size "$SIZE" --out "weights/$NAME"
echo "exported weights/$NAME; run scripts/export/make_manifest.py before uploading"
