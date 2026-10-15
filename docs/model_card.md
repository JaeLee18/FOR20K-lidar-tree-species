# Model card: multi-view ConvNeXtV2 tree-species classifier

## Overview

| | |
|---|---|
| Task | Species classification of one segmented tree from a LiDAR point cloud (33 classes) |
| Architecture | Seven orthographic geometric renders → three ConvNeXtV2-Base backbones (side / top-bottom / trunk) → max-pool per view group → concatenation with tree height → 2-layer MLP. 264.7M parameters per model |
| Released models | `convnextv2_256`: five fold models at 256 px. `convnextv2_512`: five fold models at 512 px, retrained after the paper with the same recipe ([details](../weights/README.md#available-models)) |
| Training data | FOR-species20K public labeled set: 17,707 trees, 25 source datasets, TLS / MLS / ULS |
| Initialisation | `timm/convnextv2_base.fcmae_ft_in22k_in1k` (FCMAE self-supervised, ImageNet-22k → 1k fine-tuned) |
| Objective | Logit-adjusted cross-entropy (τ = 1), best-validation-macro-F1 checkpoint |
| License | GPL-3.0-or-later (code); trained on GPL-3.0-or-later data |

## Intended use

- Research on LiDAR tree-species recognition and benchmarking on FOR-species20K.
- A strong baseline and a reference for the cross-source evaluation protocol.
- Exploratory species labelling of segmented trees from sensors and regions similar to the training sources,
  checked against local reference data.

Not intended for unsupervised operational inventories in new regions or with new sensors without local validation,
or for species outside the 33 classes. There is no rejection or "unknown" class.

## Performance

| Evaluation | Metric | Value |
|---|---|---|
| Official hidden test (species-balanced), 256 px | OA / P / R / F1 | 84.0 / 87.0 / 82.0 / 83.0 % |
| Official hidden test, 256 + 512 px | OA / P / R / F1 | 85.0 / 89.0 / 83.0 / 85.0 % |
| Public set, 5-fold OOF, 256 px | OA / macro-F1 / balanced acc. | 0.886 / 0.843 / 0.846 |
| Public set by LiDAR type, 256 px | TLS / MLS / ULS OA | 0.879 / 0.934 / 0.894 |
| Source-species holdout, 256 px (5,566 trees) | tree-weighted acc. (random split → holdout) | 0.948 → 0.602 |

## Known failure modes

- **New sources.** Accuracy on (source, species) pairs absent from training drops by 35 points (tree-weighted) even
  when the species is well represented through other sources of the same LiDAR type. The predicted probabilities
  also carry information about which dataset a scan came from (69% source accuracy from the probabilities alone).
- **Dense mixed deciduous forest.** In the BlueCat_a TLS dataset (3,474 trees), out-of-fold accuracy is 73.6% and
  this one dataset contributes about half of all public-set errors. The errors are mostly between co-occurring
  broadleaves (Acer, Carpinus, Tilia, Ulmus, Crataegus, Fraxinus, Euonymus), with frequent classes such as
  *Acer campestre* and *Carpinus betulus* absorbing rare ones.
- **Sparse drone scans.** ULS trees have far fewer points (tens of thousands versus millions for TLS) and score
  lowest on the hidden test (79.2% for the multi-resolution model).
- **Calibration.** Confidences on trees from the training sources are often near 1.0 and are not reliable as
  correctness estimates on new sources.

## Inputs and preprocessing

The input is XYZ in metres with Z up, for one tree. The cloud is centred on the median XY and shifted so the lowest
point is at Z = 0. Tree height is the Z extent. Intensity, returns and colour are not used by the image model.

## Ethical and environmental notes

The 33 species are mostly European, with some North American and Australian species, and the dataset over-represents
some sites (BlueCat_a, Weiser_2022b). Training one 256 px fold takes about 6 GPU-hours on an RTX 4090 (about 24 h at
512 px).
