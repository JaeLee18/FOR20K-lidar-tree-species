# Inference

Three ways to classify scans: the `predict.sh` wrapper, the `mvtree` command, and the Python API. All three use the
same model and give the same probabilities. For a browser or phone interface, see the [web app](web_app.md) and the
[Android app](../android/README.md).

## Before you start

- **Install** the package ([Installation](installation.md)).
- **Weights.** `predict.sh` downloads the weights on first use. Otherwise run `python -m mvtree download`
  ([weights/README.md](../weights/README.md)).
- **Resolution.** Every installed resolution is averaged: with the full download that is 256 and 512 px, the
  paper's best model. `--resolution 256` runs the faster single-resolution model.

## Input requirements

- **One file = one segmented tree.** The model classifies whole individual trees, so split plot- or tile-level
  scans into single trees first.
- **Metres, Z up.** Any horizontal offset is fine because the cloud is re-centred. The height above the lowest point
  is used as the tree height.
- **Formats:** `.las` `.laz` `.ply` `.pcd` `.xyz` `.txt` `.csv` `.pts` (first three columns = XYZ), and `.npy`
  `.npz`.
- **Species:** the 33 FOR-species20K classes listed in [`mvtree/constants.py`](../mvtree/constants.py). A tree of
  another species is mapped to the closest of these. There is no "unknown" class.
- **New sources lower accuracy.** Scans from a sensor or site that is not in FOR-species20K are the cross-source
  setting the paper studies; see the [model card](model_card.md#known-failure-modes).

## `predict.sh`

```bash
./predict.sh tree.laz                                   # one tree
./predict.sh scans/ -o predictions.csv                  # every scan in a folder (recursive) -> CSV
./predict.sh a.laz b.ply c.xyz -o out.json -k 3         # mixed formats -> JSON with all probabilities
./predict.sh tree.laz --device cpu --folds 0 --tta 1    # fast CPU preview: one model, no augmentation
./predict.sh tree.laz --save-renders renders/           # also save the 7 views and a preview montage
./predict.sh --docker tree.laz                          # run inside the Docker image
```

`predict.sh` finds the environment (`.venv`, `uv`, or the current Python), downloads missing weights, and then
calls `mvtree predict` with your options:

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | | write `.csv` (top-k columns) or `.json` (all 33 probabilities, per resolution) |
| `-k, --topk` | 5 | number of species to report |
| `-r, --resolution` | `auto` | `256`, `512`, `multi`, or `auto` (every installed resolution) |
| `--tta` | 20 | augmented passes per model (random flips and side-view rotation); `1` turns it off |
| `--folds` | all | subset of the five fold models, such as `0` or `0,2` |
| `--device` | `auto` | `cuda`, `cuda:1`, or `cpu` |
| `--low-memory` | off | keep one network in memory at a time (GPUs under 8 GB, or a CPU with little RAM) |
| `--max-batch` | 20 | augmented copies per forward pass; lower it if the GPU runs out of memory |
| `--seed` | 0 | seed for the augmentation draws, so outputs are deterministic |
| `--save-renders DIR` | | also write the seven rendered views and a preview montage |
| `--weights DIR` | `./weights` | weights folder (or set `$MVTREE_WEIGHTS`) |

**CSV columns:** `file, n_points, tree_height_m, species, genus, confidence, top2_species, top2_prob, ...`

## `mvtree` command

Installing the package adds an `mvtree` command (also available as `python -m mvtree`):

```bash
mvtree predict scans/ -o out.csv        # same options as predict.sh
mvtree render tree.laz -o renders/      # only the seven orthographic views
mvtree serve                            # self-hosted web app (see web_app.md)
mvtree download                         # weights from Google Drive, checksum-verified
mvtree info                             # versions, GPU, installed weights
```

## Python API

```python
from mvtree.predict import Predictor
from mvtree.io import read_point_cloud

predictor = Predictor()                          # finds ./weights; optional: resolutions=[256], folds=[0], tta=20
result = predictor.predict_file("tree.laz")
print(result.species, result.confidence)         # 'Pinus_sylvestris', 0.9998
print(result.top(3))                             # [('Pinus_sylvestris', 0.9998), ...]

xyz = read_point_cloud("tree.ply")               # or any (N, 3) array in metres
result = predictor.predict(xyz)
result.probs                                     # (33,) probabilities, order = mvtree.SPECIES
```
