# Model weights

Weights are not stored in git. Download them into this folder with

```bash
python -m mvtree download            # Google Drive folder from manifest.json, SHA-256 verified
python -m mvtree download --verify-only
```

`./predict.sh` runs the download automatically the first time. You can also download the Google Drive folder linked
in `manifest.json` by hand and place its contents here:

```
weights/
├── manifest.json     download link and checksums (in git)
├── convnextv2_256/   config.json, fold0.safetensors ... fold4.safetensors   (5 x 530 MB, fp16)
├── convnextv2_512/   same layout                                            (5 x 530 MB, fp16)
└── geometry_lgbm/    config.json, fold0.txt ... fold4.txt                   (optional, 87 MB)
```

`config.json` in each folder records the backbone, render size, normalisation statistics, class order, and the
validation scores of every fold.

## Available models

| Model | Render size | Folds | Public-set OOF OA / macro-F1 | Hidden test OA | Size |
|---|---|---|---|---|---|
| `convnextv2_256` | 256 px | 5 | 0.886 / 0.843 | 84.0%\* | 2.6 GB (fp16) |
| `convnextv2_512` | 512 px | 5 | 0.895 / 0.856† | n/a | 2.6 GB (fp16) |
| `convnextv2_256` + `convnextv2_512` | both | 10 | 0.899 / 0.861† | **85.0%\*†** | 5.3 GB |
| `geometry_lgbm` (optional) | n/a | 5 | 0.677 OA | n/a | 87 MB |

\* The leaderboard files also blended LightGBM probabilities on 105 handcrafted geometric features (weights 0.35 and
0.15). On the public labeled set this changes accuracy by less than 0.001. See
[docs/leaderboard.md](../docs/leaderboard.md) to reproduce the submitted files exactly.

† The 512 px weights were retrained after the paper because the original checkpoints were not kept (same recipe,
`configs/convnextv2_512.yaml`). The paper reports 0.894 / 0.855 for 512 px and 0.899 / 0.860 for both resolutions.
The 85.0% entry used the original 512 px weights; with the retrained ones, 97.4% of the hidden-test predictions
(2,196 of 2,255) are unchanged.

The predictor averages every resolution it finds, so with both folders present it runs the paper's best model. Use
`--resolution 256` for the faster single-resolution model. To train the weights yourself:
`make train-all RES=512` (about 24 GPU-hours per fold on an RTX 4090).

## For maintainers: publishing new weights

```bash
python scripts/export/export_weights.py --ckpt-dir runs/convnextv2_512 --img-size 512 --out weights/convnextv2_512
python scripts/export/make_manifest.py --gdrive-folder "https://drive.google.com/drive/folders/<FOLDER_ID>"
```

1. Upload the `convnextv2_256/`, `convnextv2_512/`, and `geometry_lgbm/` folders into one Google Drive folder,
   keeping the subfolder names.
2. Share that folder as "Anyone with the link: Viewer".
3. Put its link in `manifest.json` (`gdrive_folder`) with the command above, then commit `manifest.json`.
