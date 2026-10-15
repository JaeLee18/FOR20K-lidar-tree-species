import os

import pandas as pd
import pytest

from mvtree.splits import load_split, to_frame


def frame(root, name):
    return pd.read_csv(os.path.join(root, "splits", f"{name}.csv"))


def test_random_split_covers_every_tree_once(root):
    df = frame(root, "random_5fold")
    assert ((df.filter(like="fold_") == "val").sum(axis=1) == 1).all()
    assert len(df) == 17707


@pytest.mark.parametrize("name", ["source_holdout_5fold", "count_matched_5fold", "lodo_5fold"])
def test_holdout_species_keep_training_support(root, name):
    meta = pd.read_csv(os.path.join(root, "data", "metadata", "tree_metadata_dev.csv")).set_index("treeID")
    path = os.path.join(root, "splits", f"{name}.csv")
    for k in range(5):
        train, val = load_split(path, k)
        assert not set(train) & set(val)
        assert set(meta.loc[val, "species"]) <= set(meta.loc[train, "species"])


def test_source_holdout_scores_whole_source_species_cells(root):
    meta = pd.read_csv(os.path.join(root, "data", "metadata", "tree_metadata_dev.csv")).set_index("treeID")
    path = os.path.join(root, "splits", "source_holdout_5fold.csv")
    for k in range(5):
        train, val = load_split(path, k)
        held = set(zip(meta.loc[val, "dataset"], meta.loc[val, "species"]))
        trained = set(zip(meta.loc[train, "dataset"], meta.loc[train, "species"]))
        assert not held & trained, "a scored (source, species) cell leaked into training"


def test_roundtrip():
    df = to_frame([{"train_ids": [1, 2], "val_ids": [3]}, {"train_ids": [3], "val_ids": [1]}], [1, 2, 3])
    assert list(df.fold_0) == ["train", "train", "val"]
    assert list(df.fold_1) == ["val", "excluded", "train"]
