# Cross-validation splits

The exact splits used in the paper. Each CSV has one row per public labeled tree and one column per fold; a cell is
`train`, `val` (scored in that fold), or `excluded`.

| File | Used for |
|---|---|
| `random_5fold.csv` | the standard species-stratified split: model selection and random-split scores |
| `source_holdout_5fold.csv` | the cross-source gap: same species, unseen source |
| `count_matched_5fold.csv` | control: removes as many trees as the holdout, at random across sources |
| `lodo_5fold.csv` | control: leave-one-dataset-out |
| `source_groups.json` | the source datasets in each held-out group |

Definitions and evaluation: [docs/cross_source.md](../docs/cross_source.md). To regenerate the files:
`python scripts/data/make_splits.py`.
