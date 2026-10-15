# Released results

## `oof/`: out-of-fold predictions behind the paper's tables

Each `.npz` holds `treeID` (N,), `fold` (N,), `probs` (N, 33, float32, class order `species`), `species` (33,),
`split`, `backbone`, and `img_size`. Load one with `mvtree.evaluation.load_oof`.

| File | Split | Folds | Paper |
|---|---|---|---|
| `convnextv2_256_random.npz` | `random_5fold` | 0 to 4 | Table 2, Sec. 6.1 |
| `densenet201_256_random.npz` | `random_5fold` | 0 to 4 | Table 2, Sec. 6.1 |
| `dinov2_224_random.npz` | `random_5fold` | 0 to 4 | Table 2, Sec. 6.1 |
| `convnextv2_256_source_holdout.npz` | `source_holdout_5fold` | 0 to 4 | Table 3, Fig. 5 |
| `densenet201_256_source_holdout.npz` | `source_holdout_5fold` | 0 to 4 | Sec. 6.1 |
| `dinov2_224_source_holdout.npz` | `source_holdout_5fold` | 0 to 4 | Sec. 6.1 |
| `convnextv2_256_count_matched.npz` | `count_matched_5fold` | 2 | Sec. 6.1 (count-matched control) |
| `convnextv2_256_lodo.npz` | `lodo_5fold` | 0, 2 | Sec. 6.1 (leave-one-dataset-out control) |

`bash scripts/eval/reproduce_paper.sh` recomputes every public-set number from these files. The 512 px out-of-fold
predictions (Table 2 rows "512" and "multi-resolution") are not included yet.

## `leaderboard/`: files submitted to the FOR-species20K Codabench leaderboard

| File | Hidden-test OA |
|---|---|
| `convnextv2_256_lgbm_oa84.0.csv` | 84.0% |
| `multires_lgbm_oa85.0.csv` | 85.0% |

See [docs/leaderboard.md](../docs/leaderboard.md) for how they were produced.
