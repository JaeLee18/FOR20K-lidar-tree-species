"""Dataset of pre-rendered views for training and out-of-fold evaluation."""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import render_stats
from .render import DETAIL_VIEW, SIDE_VIEWS, TOPBOT_VIEWS


def tree_dir(render_root: str, tree_id) -> str:
    """Renders live in ``<root>/<5-digit zero-padded treeID>/``."""
    return os.path.join(render_root, f"{int(tree_id):05d}")


def has_renders(render_root: str, tree_id) -> bool:
    return os.path.isfile(os.path.join(tree_dir(render_root, tree_id), "side_000.png"))


class TreeRenderDataset(Dataset):
    """Loads the seven renders of each tree, grouped as the model expects.

    Augmentation keeps the depth/density/height encoding intact: a horizontal
    flip per image (p=0.5) and a cyclic shift of the four side views (p=0.5,
    a proxy for rotating the scanner about the vertical axis). No color jitter,
    vertical flips, mixup, or cutmix: each corrupts the geometric channels.

    Images are fed at their native resolution, so ``img_size`` must match the
    renders on disk (it only selects the normalisation statistics).
    """

    def __init__(self, tree_ids: Sequence, labels: Sequence[int], tree_heights: Sequence[float],
                 render_dir: str, img_size: int = 256, augment: bool = True) -> None:
        if not (len(tree_ids) == len(labels) == len(tree_heights)):
            raise ValueError("tree_ids, labels and tree_heights must have the same length")
        self.tree_ids = list(tree_ids)
        self.labels = list(labels)
        self.tree_heights = list(tree_heights)
        self.render_dir = render_dir
        self.augment = augment
        stats = render_stats(img_size)
        normalize = [transforms.ToTensor(), transforms.Normalize(stats["mean"], stats["std"])]
        self.base_transform = transforms.Compose(normalize)
        self.aug_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)] + normalize)

    def __len__(self) -> int:
        return len(self.tree_ids)

    @staticmethod
    def _load(path: str, transform) -> torch.Tensor:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"render not found: {path}")
        return transform(Image.open(path).convert("RGB"))

    def __getitem__(self, idx: int):
        view_dir = tree_dir(self.render_dir, self.tree_ids[idx])
        tf = self.aug_transform if self.augment else self.base_transform
        side = torch.stack([self._load(os.path.join(view_dir, f"{v}.png"), tf) for v in SIDE_VIEWS])
        topbot = torch.stack([self._load(os.path.join(view_dir, f"{v}.png"), tf) for v in TOPBOT_VIEWS])
        detail = self._load(os.path.join(view_dir, f"{DETAIL_VIEW}.png"), tf)
        if self.augment and np.random.rand() < 0.5:
            side = torch.roll(side, shifts=np.random.randint(0, 4), dims=0)
        height = torch.tensor([self.tree_heights[idx]], dtype=torch.float32)
        return side, topbot, detail, height, self.labels[idx]
