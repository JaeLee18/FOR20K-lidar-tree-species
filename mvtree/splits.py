"""Cross-validation split files.

A split file is a CSV with one row per tree and one column per fold::

    treeID,fold_0,fold_1,fold_2,fold_3,fold_4
    70,train,train,val,train,train

Each cell is ``train``, ``val`` (scored in that fold), or ``excluded`` (left
out of training and not scored; used only by the leave-one-dataset-out split).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def to_frame(splits: List[dict], all_ids) -> pd.DataFrame:
    """Convert ``[{'train_ids', 'val_ids'}, ...]`` into the CSV layout."""
    ids = np.asarray(sorted(int(t) for t in all_ids), dtype=np.int64)
    df = pd.DataFrame({"treeID": ids})
    for k, s in enumerate(splits):
        role = np.full(len(ids), "excluded", dtype=object)
        role[np.isin(ids, np.asarray(s["train_ids"], dtype=np.int64))] = "train"
        role[np.isin(ids, np.asarray(s["val_ids"], dtype=np.int64))] = "val"
        df[f"fold_{k}"] = role
    return df


def load_split(path: str, fold: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_ids, val_ids) for one fold of a split CSV."""
    df = pd.read_csv(path)
    col = f"fold_{fold}"
    if col not in df.columns:
        raise KeyError(f"{path} has no {col}")
    return (df.loc[df[col] == "train", "treeID"].to_numpy(),
            df.loc[df[col] == "val", "treeID"].to_numpy())


def n_folds(path: str) -> int:
    return sum(c.startswith("fold_") for c in pd.read_csv(path, nrows=1).columns)
