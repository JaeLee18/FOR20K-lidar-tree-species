#!/usr/bin/env python3
"""Build the cross-validation splits used in the paper.

    random_5fold          species-stratified 5-fold split (the benchmark's
                          protocol; used for model selection and the
                          random-split out-of-fold scores)
    source_holdout_5fold  source-species holdout (Sec. 4): the 25 source datasets
                          are split into 5 groups balanced by tree count; fold k
                          scores the trees of group G_k whose species also occur
                          in a source outside G_k, and trains on everything else
    count_matched_5fold   control: for every (species, LiDAR type) cell scored in
                          source_holdout fold k, remove the same number of trees
                          of that cell uniformly at random across all sources
    lodo_5fold            control: same scored trees as source_holdout, but every
                          tree of the held-out source datasets is removed from
                          training (leave-one-dataset-out)

The files shipped in ``splits/`` were produced by this script with seed 42;
re-running it reproduces them exactly.

    python scripts/data/make_splits.py --metadata data/metadata/tree_metadata_dev.csv --out splits
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.splits import to_frame  # noqa: E402

N_FOLDS = 5
SEED = 42


def random_split(m: pd.DataFrame):
    m = m.dropna(subset=["species"]).reset_index(drop=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    ids, sp = m["treeID"].values, m["species"].values
    return [{"train_ids": ids[tr], "val_ids": ids[va]} for tr, va in skf.split(ids, sp)]


def source_groups(m: pd.DataFrame):
    """Greedy balance: largest dataset first into the currently smallest group."""
    groups, loads = [[] for _ in range(N_FOLDS)], [0] * N_FOLDS
    for ds, n in sorted(m["dataset"].value_counts().to_dict().items(), key=lambda kv: -kv[1]):
        g = int(np.argmin(loads))
        groups[g].append(ds)
        loads[g] += n
    return groups


def source_holdout_split(m: pd.DataFrame, groups):
    sp_datasets = m.groupby("species")["dataset"].agg(set).to_dict()
    multi_source = {s for s, d in sp_datasets.items() if len(d) >= 2}
    splits = []
    for held in map(set, groups):
        eligible = {s for s in multi_source if sp_datasets[s] - held}
        is_val = m["dataset"].isin(held) & m["species"].isin(eligible)
        val_ids, train_ids = m.loc[is_val, "treeID"].values, m.loc[~is_val, "treeID"].values
        assert set(m.loc[is_val, "species"]) <= set(m.loc[~is_val, "species"])
        splits.append({"train_ids": train_ids, "val_ids": val_ids})
    return splits


def count_matched_split(m: pd.DataFrame, holdout):
    rng = np.random.RandomState(SEED)
    pool = m.groupby(["species", "data_type"])["treeID"].agg(list).to_dict()
    splits = []
    for s in holdout:
        val_a = set(int(t) for t in s["val_ids"])
        cells = m[m.treeID.isin(val_a)].groupby(["species", "data_type"]).size().to_dict()
        val_b = []
        for cell, n in sorted(cells.items()):
            cand = np.asarray(pool[cell], dtype=np.int64)
            val_b.extend(int(t) for t in (cand.copy() if len(cand) <= n else rng.choice(cand, size=n, replace=False)))
        val_b = np.array(sorted(set(val_b)), dtype=np.int64)
        splits.append({"train_ids": m.loc[~m.treeID.isin(val_b), "treeID"].values, "val_ids": val_b})
    return splits


def lodo_split(m: pd.DataFrame, holdout):
    splits = []
    for s in holdout:
        val_ids = np.asarray([int(t) for t in s["val_ids"]], dtype=np.int64)
        held = set(m.loc[m.treeID.isin(val_ids), "dataset"])
        train_ids = m.loc[~m["dataset"].isin(held), "treeID"].values
        splits.append({"train_ids": train_ids, "val_ids": val_ids})
    return splits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="data/metadata/tree_metadata_dev.csv")
    ap.add_argument("--out", default="splits")
    args = ap.parse_args()

    m = pd.read_csv(args.metadata)
    os.makedirs(args.out, exist_ok=True)
    groups = source_groups(m)
    holdout = source_holdout_split(m, groups)
    schemes = {
        "random_5fold": random_split(m),
        "source_holdout_5fold": holdout,
        "count_matched_5fold": count_matched_split(m, holdout),
        "lodo_5fold": lodo_split(m, holdout),
    }
    for name, splits in schemes.items():
        df = to_frame(splits, m["treeID"])
        df.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        counts = ", ".join(f"{(df[c] == 'val').sum()}" for c in df.columns[1:])
        print(f"{name:22s} val per fold: {counts}")
    with open(os.path.join(args.out, "source_groups.json"), "w") as fh:
        json.dump({f"fold_{k}": g for k, g in enumerate(groups)}, fh, indent=2)
    print(f"wrote {args.out}/")


if __name__ == "__main__":
    main()
