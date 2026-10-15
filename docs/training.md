# Training and reproducing the paper

## Reproducing the paper

**From the released predictions (about a minute on CPU).** [`results/oof/`](../results/oof) holds the out-of-fold
probabilities of every model in the paper's tables, except the 512 px runs, which will be added with the 512 px
weights.

```bash
make paper-tables        # same as: bash scripts/eval/reproduce_paper.sh
```

This prints the ablation table, the cross-source gap for all three backbones with both controls, and the
source-recovery results. All numbers match the paper. CI runs the same script on every push.

**From scratch.** These commands go from the raw data to a leaderboard submission. Each step is explained in the
sections below.

```bash
python scripts/data/download_data.py                   # FOR-species20K from Zenodo (27 GB, md5-checked)
make renders RES=256                                   # 7 views x 19,962 trees, read straight from the zips
bash scripts/train/train_all_folds.sh configs/convnextv2_256.yaml runs/convnextv2_256
python scripts/eval/evaluate_oof.py runs/convnextv2_256
python scripts/eval/predict_testset.py --source data/raw/test.zip --out submissions/convnextv2_256.csv
```

| Step | Hardware | Time |
|---|---|---|
| Render all 19,962 trees at 256 or 512 px | 16 CPU cores | about 20 to 30 min |
| Train one fold, 256 px | 1× RTX 4090 (24 GB) | about 6 h |
| Train one fold, 512 px | 1× RTX 4090 (24 GB) | about 24 h |
| Predict 2,255 test trees (5 folds × 20 TTA, 256 px) | 1× RTX 4090 | about 1.5 h |

Everything below runs on one 24 GB GPU (developed on an RTX 4090). Commands assume the repository root and an
environment with the training extras (`uv sync --extra geometry --extra train`).

## 1. Data

```bash
python scripts/data/download_data.py                 # data/raw/{dev,test}.zip + data/metadata/tree_metadata_dev.csv
```

The public labeled set has 17,707 trees; the hidden test has 2,255 scans whose labels are only scored on
[Codabench](https://www.codabench.org/). The zips never need to be extracted.

## 2. Renders

```bash
python scripts/data/render_dataset.py --source data/raw/dev.zip  --out data/renders_256/dev  --size 256
python scripts/data/render_dataset.py --source data/raw/test.zip --out data/renders_256/test --size 256
# second resolution for the multi-resolution model
python scripts/data/render_dataset.py --source data/raw/dev.zip  --out data/renders_512/dev  --size 512
```

Each tree becomes `data/renders_<size>/<split>/<5-digit treeID>/{side_000,side_090,side_180,side_270,top,bottom,detail}.png`.
The renderer (`mvtree/render.py`) is a vectorised version of the per-point loop the paper's renders were made with;
`tests/test_render.py` checks that the two produce identical images. Rendering is resumable: finished trees are
skipped.

The channel statistics in `mvtree/constants.py::RENDER_STATS` were computed over all public labeled renders.
Recompute them only for a new resolution:

```bash
python scripts/data/compute_render_stats.py --renders data/renders_384/dev
```

## 3. Splits

`splits/` ships the exact splits used in the paper; `python scripts/data/make_splits.py` regenerates them byte-identically
(seed 42). See [the cross-source protocol](cross_source.md) for what each split holds out.

## 4. Train

```bash
# one fold
python scripts/train/train.py --config configs/convnextv2_256.yaml --fold 0 --out runs/convnextv2_256
# all five folds, then export to weights/convnextv2_256/
bash scripts/train/train_all_folds.sh configs/convnextv2_256.yaml runs/convnextv2_256
bash scripts/train/train_all_folds.sh configs/convnextv2_512.yaml runs/convnextv2_512
```

| Config | Backbone | Batch × accum | Time / fold (RTX 4090) |
|---|---|---|---|
| `convnextv2_256.yaml` | ConvNeXtV2-Base ×3 (about 19 GB peak) | 8 × 4 | about 6 h |
| `convnextv2_512.yaml` | ConvNeXtV2-Base ×3 | 2 × 16 | about 24 h |
| `dinov2_224.yaml` | DINOv2-Base ×3 (renders resized to 224) | 16 × 2 | about 2 h |
| `densenet201_256.yaml` | DenseNet-201 ×3 | 8 × 4 | |

The recipe in the paper (Supp. Table A1):

| Setting | Value |
|---|---|
| Loss | logit-adjusted cross-entropy, τ = 1, prior from the fold's training labels, no label smoothing |
| Sampling | plain shuffling (no class-balanced sampler; the loss already corrects for the prior) |
| Optimiser | AdamW, weight decay 1e-4 |
| Schedule | epochs 1 to 3: head only, LR 3e-4, cosine; epochs 4 to 40: all weights, backbone 3e-5 / head 3e-4, cosine to 1% |
| Batch | effective 32, mixed precision, gradient clipping 1.0 |
| Augmentation | horizontal flip per view (p = 0.5), cyclic shift of the four side views (p = 0.5) |
| Model selection | epoch with the best validation macro-F1 |

Notes:
- `--resume` continues from `fold{k}_last.pt`. Re-running `train_all_folds.sh` skips finished folds.
- `--wandb` logs to Weights & Biases (off by default).
- The paper's runs were not seeded; `--seed` makes new runs repeatable but will not reproduce the released
  checkpoints bit for bit. For reference, the five released 256 px folds span 0.882 to 0.890 validation OA.
- The DenseNet-201 row of Table 2 comes from the original DetailView-style recipe (class-weighted cross-entropy with
  a class-balanced sampler and label smoothing 0.1). `configs/densenet201_256.yaml` uses the ConvNeXtV2 recipe, which
  is what the DenseNet cross-source runs used. The paper's DenseNet out-of-fold predictions are in `results/oof/`.

Each fold writes `best_fold{k}.pt`, `oof_fold{k}.npz` (validation probabilities of the best epoch),
`fold{k}_history.csv`, and `fold{k}.log`.

## 5. Evaluate

```bash
python scripts/eval/evaluate_oof.py runs/convnextv2_256                                     # one model
python scripts/eval/evaluate_oof.py runs/convnextv2_256 runs/convnextv2_512 --ensemble      # + multi-resolution row
python scripts/eval/evaluate_oof.py runs/convnextv2_256 --per-species
```

## 6. Export and predict

```bash
python scripts/export/export_weights.py --ckpt-dir runs/convnextv2_512 --img-size 512 --out weights/convnextv2_512
python scripts/export/make_manifest.py            # refresh checksums before uploading
python scripts/eval/predict_testset.py --source data/raw/test.zip --out submissions/multires.csv
```

`predict.sh` and `Predictor` pick up any `weights/convnextv2_<size>/` folder automatically.

## 7. Optional: geometric-feature LightGBM

The leaderboard files blended a LightGBM model on 105 handcrafted features (crown hull, vertical density profile,
height percentiles, eigenvalue shape descriptors, intensity and return statistics, trunk and crown part features).
`mvtree/geometry.py` holds the feature code; the released boosters are in `weights/geometry_lgbm/`. See
[leaderboard.md](leaderboard.md).
