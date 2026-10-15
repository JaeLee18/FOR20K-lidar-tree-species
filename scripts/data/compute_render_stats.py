#!/usr/bin/env python3
"""Per-channel mean/std of the renders over the public labeled set.

The released statistics live in ``mvtree/constants.py::RENDER_STATS``; rerun
this only for a new render resolution or a new dataset.

    python scripts/data/compute_render_stats.py --renders data/renders_256/dev
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.render import VIEW_NAMES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--renders", required=True)
    args = ap.parse_args()

    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    n_pix = 0
    trees = sorted(os.listdir(args.renders))
    for tid in tqdm(trees):
        for v in VIEW_NAMES:
            path = os.path.join(args.renders, tid, f"{v}.png")
            if not os.path.isfile(path):
                continue
            x = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64).reshape(-1, 3) / 255.0
            total += x.sum(0)
            total_sq += (x ** 2).sum(0)
            n_pix += x.shape[0]
    mean = total / n_pix
    std = np.sqrt(total_sq / n_pix - mean ** 2)
    print(f"{len(trees)} trees, {n_pix:,} pixels per channel")
    print(f'"mean": [{", ".join(f"{m:.4f}" for m in mean)}], "std": [{", ".join(f"{s:.4f}" for s in std)}]')


if __name__ == "__main__":
    main()
