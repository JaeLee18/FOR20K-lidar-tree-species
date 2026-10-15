"""Out-of-fold (OOF) prediction files and the metrics reported in the paper.

An OOF file is an ``.npz`` with ``treeID`` (N,), ``fold`` (N,), ``probs`` (N, 33)
and ``species`` (33,). ``scripts/train/train.py`` writes one per fold
(``oof_fold{k}.npz``); ``results/oof/*.npz`` hold the pooled files behind the
paper's tables.
"""

from __future__ import annotations

import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from .constants import PLATFORMS, SPECIES


def load_oof(path: str) -> pd.DataFrame:
    """Load an OOF ``.npz`` file, or every ``oof_fold*.npz`` in a run directory.

    Returns a frame with columns treeID, fold, pred, and p_0 ... p_32.
    """
    files = sorted(glob.glob(os.path.join(path, "oof_fold*.npz"))) if os.path.isdir(path) else [path]
    if not files:
        raise FileNotFoundError(f"no OOF files in {path}")
    frames = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        if list(z["species"]) != SPECIES:
            raise ValueError(f"{f}: class order differs from mvtree.constants.SPECIES")
        df = pd.DataFrame(z["probs"], columns=[f"p_{i}" for i in range(len(SPECIES))])
        df.insert(0, "fold", z["fold"].astype(int))
        df.insert(0, "treeID", z["treeID"].astype(int))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df.treeID.duplicated().any():
        raise ValueError(f"{path}: a tree is scored in more than one fold")
    df["pred"] = probs_of(df).argmax(1)
    return df


def probs_of(df: pd.DataFrame) -> np.ndarray:
    return df[[f"p_{i}" for i in range(len(SPECIES))]].to_numpy()


def average_oof(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Average class probabilities of several runs over their common trees
    (the multi-resolution ensemble when given the 256 and 512 px runs)."""
    common = set(frames[0].treeID)
    for f in frames[1:]:
        common &= set(f.treeID)
    aligned = [f[f.treeID.isin(common)].sort_values("treeID").reset_index(drop=True) for f in frames]
    out = aligned[0][["treeID", "fold"]].copy()
    mean = np.mean([probs_of(f) for f in aligned], axis=0)
    for i in range(mean.shape[1]):
        out[f"p_{i}"] = mean[:, i]
    out["pred"] = mean.argmax(1)
    return out


def attach_metadata(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta[["treeID", "species", "dataset", "data_type"]].copy()
    meta["label"] = meta.species.map({s: i for i, s in enumerate(SPECIES)})
    return df.merge(meta, on="treeID", how="left", validate="one_to_one")


def summary(df: pd.DataFrame) -> Dict[str, float]:
    """Natural-distribution accuracy, macro metrics, and per-platform accuracy."""
    y, p = df.label.to_numpy(), df.pred.to_numpy()
    out = {
        "n": len(df),
        "OA": accuracy_score(y, p),
        "macro_F1": f1_score(y, p, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, p),
    }
    for plat in PLATFORMS:
        m = (df.data_type == plat).to_numpy()
        out[plat] = accuracy_score(y[m], p[m]) if m.any() else float("nan")
    return out


def per_species_f1(df: pd.DataFrame) -> pd.Series:
    y, p = df.label.to_numpy(), df.pred.to_numpy()
    f1 = f1_score(y, p, labels=list(range(len(SPECIES))), average=None, zero_division=0)
    return pd.Series(f1, index=SPECIES)
