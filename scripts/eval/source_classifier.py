#!/usr/bin/env python3
"""Can the source dataset be read off the species probabilities? (paper Sec. 6.1)

A logistic regression predicts which of the 25 source datasets each tree came
from, using only the model's 33-way out-of-fold probability vector (5-fold
stratified CV over trees). Accuracy far above chance (1/25) means the outputs
carry source identity, which a random split can reward.

    python scripts/eval/source_classifier.py                                  # ConvNeXtV2
    python scripts/eval/source_classifier.py --oof results/oof/dinov2_224_random.npz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.constants import SPECIES  # noqa: E402
from mvtree.evaluation import attach_metadata, load_oof, probs_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default="results/oof/convnextv2_256_random.npz")
    ap.add_argument("--metadata", default="data/metadata/tree_metadata_dev.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = attach_metadata(load_oof(args.oof), pd.read_csv(args.metadata))
    X = probs_of(df)
    sources = sorted(df.dataset.unique())
    y = df.dataset.map({s: i for i, s in enumerate(sources)}).to_numpy()

    pred = np.empty(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for tr, va in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed).fit(X[tr], y[tr])
        pred[va] = clf.predict(X[va])
    correct = pred == y

    n_src_per_species = df.groupby("species").dataset.nunique()
    multi = df.species.map(n_src_per_species).to_numpy() > 1
    print(f"trees: {len(df)}, source datasets: {len(sources)}, chance: {1 / len(sources):.3f}")
    print(f"source accuracy from species probabilities: {correct.mean():.3f} "
          f"({correct.mean() * len(sources):.0f}x chance)")
    print(f"  on species found in more than one source: {correct[multi].mean():.3f} (n={multi.sum()})")
    for plat in ("TLS", "MLS", "ULS"):
        m = (df.data_type == plat).to_numpy()
        print(f"  {plat}: {correct[m].mean():.3f} (n={m.sum()})")

    # Two readings of "the error stays inside the tree's source":
    #   modal:   the predicted species' most frequent source is the tree's source (paper: 63%)
    #   present: the predicted species occurs anywhere in the tree's source
    species_in_source = df.groupby("dataset").species.agg(set).to_dict()
    modal_source = df.groupby("species").dataset.agg(lambda s: s.mode().iloc[0]).to_dict()
    wrong = df[df.pred != df.label]
    modal = np.mean([modal_source[SPECIES[p]] == d for p, d in zip(wrong.pred, wrong.dataset)])
    present = np.mean([SPECIES[p] in species_in_source[d] for p, d in zip(wrong.pred, wrong.dataset)])
    print(f"misclassified trees (n={len(wrong)}) whose predicted species has the tree's source as its "
          f"most frequent source: {modal:.3f}; occurs in the tree's source at all: {present:.3f}")
    print(f"species occurring in a single source: {(n_src_per_species == 1).sum()} of {len(n_src_per_species)}")


if __name__ == "__main__":
    main()
