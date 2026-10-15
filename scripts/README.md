# Scripts

Command-line pipelines around the `mvtree` package, grouped by stage. Run them from the repository root; each
script's docstring shows its usage, and `--help` lists its options. [Training](../docs/training.md) walks through
them in order.

| Stage | Script | What it does |
|---|---|---|
| **data** | [`download_data.py`](data/download_data.py) | download FOR-species20K from Zenodo (27 GB, md5-checked) |
| | [`render_dataset.py`](data/render_dataset.py) | render the seven views of every tree in a split |
| | [`compute_render_stats.py`](data/compute_render_stats.py) | per-channel render statistics (only needed for a new resolution) |
| | [`make_splits.py`](data/make_splits.py) | regenerate `splits/` byte-identically |
| **train** | [`train.py`](train/train.py) | train one cross-validation fold |
| | [`train_all_folds.sh`](train/train_all_folds.sh) | train all five folds, evaluate, and export the weights |
| **eval** | [`evaluate_oof.py`](eval/evaluate_oof.py) | out-of-fold metrics on the public labeled set (paper Table 2) |
| | [`cross_source_eval.py`](eval/cross_source_eval.py) | the cross-source gap and its controls (paper Sec. 4 and 6.1) |
| | [`source_classifier.py`](eval/source_classifier.py) | predict the source dataset from the species probabilities |
| | [`predict_testset.py`](eval/predict_testset.py) | predict the hidden test set and write a Codabench submission |
| | [`reproduce_paper.sh`](eval/reproduce_paper.sh) | every public-set number in the paper from `results/oof/` |
| **export** | [`export_weights.py`](export/export_weights.py) | convert training checkpoints into the release format |
| | [`make_manifest.py`](export/make_manifest.py) | checksums and download link in `weights/manifest.json` |
| | [`export_mobile.py`](export/export_mobile.py) | int8 ONNX model for the [Android app](../android/README.md) |
| **media** | [`make_assets.py`](media/make_assets.py) | figures and demo animation in `docs/assets/` |
