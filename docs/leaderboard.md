# Reproducing the leaderboard submissions

The FOR-species20K hidden test (2,255 scans in `test.zip`) is scored on Codabench. It samples a similar number of
trees per species, so its overall accuracy behaves like a macro metric.

| Entry | OA | P | R | F1 | Submitted file |
|---|---|---|---|---|---|
| ConvNeXtV2×3 + LA-CE, 256 px | 84.0 | 87.0 | 82.0 | 83.0 | [`results/leaderboard/convnextv2_256_lgbm_oa84.0.csv`](../results/leaderboard/convnextv2_256_lgbm_oa84.0.csv) |
| + multi-resolution ensemble | 85.0 | 89.0 | 83.0 | 85.0 | [`results/leaderboard/multires_lgbm_oa85.0.csv`](../results/leaderboard/multires_lgbm_oa85.0.csv) |

## How the submitted files were made

Both files blend the image model with LightGBM probabilities on 105 handcrafted geometric features:

| Entry | Blend |
|---|---|
| 84.0 | 0.65 · p<sub>256</sub> + 0.35 · p<sub>LightGBM</sub> |
| 85.0 | 0.40 · p<sub>256</sub> + 0.45 · p<sub>512</sub> + 0.15 · p<sub>LightGBM</sub> |

Each p<sub>256</sub> and p<sub>512</sub> averages 5 fold models × 20 test-time augmentations. The blend weights were
tuned once on fold 0 of the public labeled set. There, LightGBM changes overall accuracy by less than 0.001, which is
why the paper describes and analyses the image model alone. The commands below reproduce the files including the blend.

```bash
python -m mvtree download                       # includes weights/geometry_lgbm
# 84.0 entry
python scripts/eval/predict_testset.py --source data/raw/test.zip --resolution 256 \
    --geometry-weight 0.35 --out submissions/convnextv2_256_lgbm.csv
# 85.0 entry (needs weights/convnextv2_512)
python scripts/eval/predict_testset.py --source data/raw/test.zip --resolution multi \
    --resolution-weights 0.40,0.45 --geometry-weight 0.15 --out submissions/multires_lgbm.csv
# image model only, as described in the paper
python scripts/eval/predict_testset.py --source data/raw/test.zip --out submissions/image_only.csv
cd submissions && zip multires_lgbm.zip multires_lgbm.csv       # Codabench expects a zipped CSV
```

The original runs drew test-time augmentations without a seed, so a fresh run matches the submitted labels on almost
every tree but not all of them. On the first 150 test scans, the released 256 px pipeline with the 0.35 blend
reproduces the 84.0 file on 150 of 150 trees, and the image model alone (no LightGBM) agrees with it on 149 of 150.

The CSV format is `treeID,predicted_species`, one row per scan.
