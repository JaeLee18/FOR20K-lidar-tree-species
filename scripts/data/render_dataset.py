#!/usr/bin/env python3
"""Render the 7 orthographic views of every tree in a split.

Reads scans from a directory of .las/.laz files or straight from the Zenodo
zip, and writes ``<out>/<5-digit treeID>/<view>.png``. Already-rendered trees
are skipped, so the script can be interrupted and resumed.

    python scripts/data/render_dataset.py --source data/raw/dev.zip  --out data/renders_256/dev  --size 256
    python scripts/data/render_dataset.py --source data/raw/test.zip --out data/renders_512/test --size 512
"""

import argparse
import io
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.render import VIEW_NAMES, render_tree  # noqa: E402

_ZIP = None


def list_scans(source: str):
    if source.endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            return sorted(n for n in zf.namelist() if n.lower().endswith((".las", ".laz")))
    return sorted(os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith((".las", ".laz")))


def _init(source: str) -> None:
    global _ZIP
    _ZIP = zipfile.ZipFile(source) if source.endswith(".zip") else None


def read_xyz(name: str) -> np.ndarray:
    import laspy

    las = laspy.read(io.BytesIO(_ZIP.read(name))) if _ZIP is not None else laspy.read(name)
    # float32 rounding matches the training renders (see mvtree.io)
    return np.stack([las.x, las.y, las.z], axis=-1).astype(np.float32).astype(np.float64)


def work(job):
    name, out_root, size = job
    tree_id = int(os.path.splitext(os.path.basename(name))[0])
    out = os.path.join(out_root, f"{tree_id:05d}")
    if all(os.path.isfile(os.path.join(out, f"{v}.png")) for v in VIEW_NAMES):
        return tree_id, "skip"
    try:
        render_tree(read_xyz(name), out, img_size=size)
        return tree_id, "ok"
    except Exception as exc:  # keep going; report at the end
        return tree_id, f"error: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="dev.zip / test.zip or a directory of .laz files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    scans = list_scans(args.source)
    os.makedirs(args.out, exist_ok=True)
    print(f"{len(scans)} scans from {args.source} -> {args.out} at {args.size}px, {args.workers} workers")
    errors = []
    with Pool(args.workers, initializer=_init, initargs=(args.source,)) as pool:
        jobs = [(s, args.out, args.size) for s in scans]
        for tid, status in tqdm(pool.imap_unordered(work, jobs, chunksize=4), total=len(jobs)):
            if status.startswith("error"):
                errors.append((tid, status))
    for tid, status in errors:
        print(f"  {tid:05d}: {status}")
    print(f"done: {len(scans) - len(errors)} ok, {len(errors)} failed")


if __name__ == "__main__":
    main()
