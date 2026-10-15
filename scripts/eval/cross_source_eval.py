#!/usr/bin/env python3
"""Cross-source evaluation: the source-species holdout gap (paper Sec. 4 and 6.1).

Each tree held out by ``splits/source_holdout_5fold.csv`` is scored twice: by
the model trained without its (source, species) cell, and by the random-split
model's out-of-fold prediction for the same tree. Only the split differs.

A (fold, species, LiDAR type) cell is scored when it keeps at least ``S``
training trees of the same LiDAR type outside the held-out group and has at
least ``H`` held-out trees (defaults 50 and 20). This separates cross-source
shift from a species simply disappearing from training.

Control arms (count-matched removal, leave-one-dataset-out) are scored on the
same eligible cells with ``--arm NAME=OOF.npz:SPLIT.csv``.

    python scripts/eval/cross_source_eval.py                                 # ConvNeXtV2 (shipped OOF)
    python scripts/eval/cross_source_eval.py \
        --random results/oof/dinov2_224_random.npz \
        --holdout results/oof/dinov2_224_source_holdout.npz             # DINOv2
    python scripts/eval/cross_source_eval.py \
        --arm count_matched=results/oof/convnextv2_256_count_matched.npz:splits/count_matched_5fold.csv \
        --arm lodo=results/oof/convnextv2_256_lodo.npz:splits/lodo_5fold.csv
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.constants import SPECIES  # noqa: E402
from mvtree.evaluation import load_oof  # noqa: E402
from mvtree.splits import load_split, n_folds  # noqa: E402

KEY = ["fold", "species", "data_type"]


def arm_records(oof: pd.DataFrame, split_csv: str, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-tree correctness plus the cell's training support under this arm."""
    label = meta.set_index("treeID").species.map({s: i for i, s in enumerate(SPECIES)})
    rows = []
    for k in sorted(oof.fold.unique()):
        train_ids, val_ids = load_split(split_csv, k)
        held = meta[meta.treeID.isin(val_ids)]
        support = meta[~meta.treeID.isin(val_ids)].groupby(["species", "data_type"]).size()
        held_n = held.groupby(["species", "data_type"]).size()
        o = oof[oof.fold == k].merge(held[["treeID", "species", "data_type", "dataset"]], on="treeID")
        o["ok"] = (o.pred.values == label.loc[o.treeID].values).astype(int)
        o["supp"] = [support.get((s, p), 0) for s, p in zip(o.species, o.data_type)]
        o["held_n"] = [held_n.get((s, p), 0) for s, p in zip(o.species, o.data_type)]
        rows.append(o[["treeID", "fold", "species", "data_type", "dataset", "ok", "supp", "held_n"]])
    return pd.concat(rows, ignore_index=True)


def headline(e: pd.DataFrame) -> dict:
    per_sp = e.groupby("species").agg(random=("rand_ok", "mean"), holdout=("ok", "mean"), n=("ok", "size"))
    by_fold = e.groupby("fold")[["rand_ok", "ok"]].mean()
    per_fold = by_fold.rand_ok - by_fold.ok
    return dict(n=len(e), n_species=e.species.nunique(), rand=e.rand_ok.mean(), hold=e.ok.mean(),
                rand_macro=per_sp.random.mean(), hold_macro=per_sp.holdout.mean(),
                per_sp=per_sp.assign(drop=per_sp.random - per_sp.holdout), per_fold=per_fold)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--random", default="results/oof/convnextv2_256_random.npz")
    ap.add_argument("--holdout", default="results/oof/convnextv2_256_source_holdout.npz")
    ap.add_argument("--holdout-split", default="splits/source_holdout_5fold.csv")
    ap.add_argument("--metadata", default="data/metadata/tree_metadata_dev.csv")
    ap.add_argument("-S", "--min-support", type=int, default=50)
    ap.add_argument("-H", "--min-heldout", type=int, default=20)
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=OOF:SPLIT")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    rand_pred = load_oof(args.random).set_index("treeID").pred
    hold = arm_records(load_oof(args.holdout), args.holdout_split, meta)
    hold["rand_ok"] = (rand_pred.loc[hold.treeID].values ==
                       hold.species.map({s: i for i, s in enumerate(SPECIES)}).values).astype(int)
    print(f"held-out trees: {len(hold)} over folds {sorted(int(f) for f in hold.fold.unique())} "
          f"(of {n_folds(args.holdout_split)})")

    def eligible(S, H):
        return hold[(hold.supp >= S) & (hold.held_n >= H)]

    e = eligible(args.min_support, args.min_heldout)
    h = headline(e)
    print(f"\n== support-controlled gap (>= {args.min_support} same-type training trees, "
          f">= {args.min_heldout} held-out trees per cell) ==")
    print(f"scored: {h['n']} trees, {h['n_species']} species")
    print(f"species mean (unweighted): random {h['rand_macro']:.3f} -> holdout {h['hold_macro']:.3f} "
          f"(drop {h['rand_macro'] - h['hold_macro']:.3f})")
    print(f"tree-weighted:             random {h['rand']:.3f} -> holdout {h['hold']:.3f} "
          f"(drop {h['rand'] - h['hold']:.3f})")
    print("per-fold drop (tree-weighted): " + ", ".join(f"fold {k}: {v:.3f}" for k, v in h["per_fold"].items()))

    d = h["per_sp"]["drop"]
    print("\nleave-one-species-out species-mean drop: "
          f"{min(d.drop(s).mean() for s in d.index):.3f} to {max(d.drop(s).mean() for s in d.index):.3f}")

    print("\nper species (pooled over folds):")
    print(h["per_sp"].sort_values("drop", ascending=False).to_string(float_format=lambda x: f"{x:.3f}"))

    print("\nthreshold sweep:")
    for S in (30, 50, 80):
        hs = headline(eligible(S, args.min_heldout))
        print(f"  S>={S:<3d} {hs['n_species']:>2d} species {hs['n']:>5d} trees | species-mean drop "
              f"{hs['rand_macro'] - hs['hold_macro']:.3f} | tree-weighted drop {hs['rand'] - hs['hold']:.3f}")

    if args.arm:
        cells = e[KEY].drop_duplicates()
        print("\ncontrol arms on the same eligible cells (accuracy):")
        for spec in args.arm:
            name, rest = spec.split("=", 1)
            oof_path, split_csv = rest.rsplit(":", 1)
            a = arm_records(load_oof(oof_path), split_csv, meta).merge(cells, on=KEY)
            for k in sorted(a.fold.unique()):
                ek = e[e.fold == k]
                print(f"  fold {k}: random {ek.rand_ok.mean():.3f} | source-holdout {ek.ok.mean():.3f} "
                      f"(n={len(ek)}) | {name} {a[a.fold == k].ok.mean():.3f} (n={(a.fold == k).sum()})")
            ep = e[e.fold.isin(a.fold.unique())]
            print(f"  {name}, pooled over folds {sorted(int(f) for f in a.fold.unique())}: drop from random "
                  f"{ep.rand_ok.mean() - ep.ok.mean():.3f} (source-holdout) vs "
                  f"{ep.rand_ok.mean() - a.ok.mean():.3f} ({name})")


if __name__ == "__main__":
    main()
