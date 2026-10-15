#!/usr/bin/env bash
# Reproduce the paper's public-labeled-set numbers from the released
# out-of-fold predictions in results/oof (CPU only, about a minute).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
hr() { printf '\n%s\n%s\n' "$1" "$(printf '=%.0s' $(seq 1 ${#1}))"; }

hr "Table 2: backbone ablation, 5-fold out-of-fold (random split)"
$PY scripts/eval/evaluate_oof.py results/oof/densenet201_256_random.npz \
    results/oof/dinov2_224_random.npz results/oof/convnextv2_256_random.npz

hr "Sec. 4 / 6.1 / Table 3: source-species holdout, ConvNeXtV2 (+ count-matched and LODO controls)"
$PY scripts/eval/cross_source_eval.py \
    --arm count_matched=results/oof/convnextv2_256_count_matched.npz:splits/count_matched_5fold.csv \
    --arm lodo=results/oof/convnextv2_256_lodo.npz:splits/lodo_5fold.csv

for B in densenet201_256 dinov2_224; do
  hr "Sec. 6.1: source-species holdout, $B"
  $PY scripts/eval/cross_source_eval.py --random results/oof/${B}_random.npz \
      --holdout results/oof/${B}_source_holdout.npz | sed -n '1,7p'
done

hr "Sec. 6.1: source dataset recovered from species probabilities"
for B in convnextv2_256 dinov2_224; do
  echo "-- $B"
  $PY scripts/eval/source_classifier.py --oof results/oof/${B}_random.npz
done
