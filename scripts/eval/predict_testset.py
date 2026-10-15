#!/usr/bin/env python3
"""Predict the FOR-species20K hidden test set and write a Codabench submission.

    python scripts/eval/predict_testset.py --source data/raw/test.zip --out submissions/multires.csv
    # reproduce the leaderboard files exactly (ConvNeXtV2 + LightGBM blend):
    python scripts/eval/predict_testset.py --source data/raw/test.zip --geometry-weight 0.35 \
        --resolution 256 --out submissions/convnextv2_256_lgbm.csv          # 84.0 entry
    python scripts/eval/predict_testset.py --source data/raw/test.zip --geometry-weight 0.15 \
        --resolution multi --out submissions/multires_lgbm.csv              # 85.0 entry

The output CSV has columns ``treeID,predicted_species``; zip it for upload.
A ``.npz`` with the full probability matrix is written next to it.

Blend: p = (1 - w) * p_image + w * p_lightgbm, where p_image averages the
resolutions equally. The 85.0 entry used (p256, p512, p_lgbm) = (0.40, 0.45,
0.15); pass ``--resolution-weights 0.40,0.45`` to match it to the digit.
"""

import argparse
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.constants import SPECIES  # noqa: E402
from mvtree.predict import Predictor  # noqa: E402


def iter_scans(source: str):
    """Yield (treeID, raw bytes or path) for every .las/.laz in a zip or directory."""
    if source.endswith(".zip"):
        with zipfile.ZipFile(source) as zf:
            for name in sorted(n for n in zf.namelist() if n.lower().endswith((".las", ".laz"))):
                yield int(os.path.splitext(os.path.basename(name))[0]), zf.read(name)
    else:
        for name in sorted(f for f in os.listdir(source) if f.lower().endswith((".las", ".laz"))):
            yield int(os.path.splitext(name)[0]), os.path.join(source, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="data/raw/test.zip")
    ap.add_argument("--out", default="submissions/predictions.csv")
    ap.add_argument("--resolution", default="auto", choices=["auto", "256", "512", "multi"])
    ap.add_argument("--resolution-weights", help="e.g. 0.40,0.45 for (256, 512); default equal")
    ap.add_argument("--geometry-weight", type=float, default=0.0, help="LightGBM blend weight (0 = image only)")
    ap.add_argument("--tta", type=int, default=20)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: first N scans only")
    args = ap.parse_args()

    import laspy

    res = None if args.resolution == "auto" else [256, 512] if args.resolution == "multi" else [int(args.resolution)]
    predictor = Predictor(weights=args.weights, resolutions=res, device=args.device, tta=args.tta)
    rw = None
    if args.resolution_weights:
        rw = np.array([float(x) for x in args.resolution_weights.split(",")])
        rw = rw / rw.sum()
        assert len(rw) == len(predictor.resolutions), "one weight per resolution"
    boosters = None
    if args.geometry_weight > 0:
        from mvtree.geometry import extract_features, load_boosters, predict_proba
        from mvtree.weights import weights_root
        boosters, feature_cols = load_boosters(os.path.join(weights_root(args.weights), "geometry_lgbm"))
    print(f"model: {predictor.describe()}, LightGBM weight {args.geometry_weight}")

    ids, probs = [], []
    scans = list(iter_scans(args.source))
    if args.limit:
        scans = scans[:args.limit]
    for tid, data in tqdm(scans, desc="test trees"):
        las = laspy.read(io.BytesIO(data) if isinstance(data, bytes) else data)
        xyz = np.stack([las.x, las.y, las.z], axis=-1)
        pred = predictor.predict(xyz)
        if rw is not None:
            p = np.sum([w * pred.per_resolution[r] for w, r in zip(rw, predictor.resolutions)], axis=0)
        else:
            p = pred.probs
        if boosters is not None:
            feats = extract_features(xyz, np.asarray(las.intensity), np.asarray(las.return_number))
            p = (1 - args.geometry_weight) * p + args.geometry_weight * predict_proba(feats, boosters, feature_cols)
        ids.append(tid)
        probs.append(p)

    probs = np.asarray(probs, dtype=np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    labels = [SPECIES[i] for i in probs.argmax(1)]
    pd.DataFrame({"treeID": ids, "predicted_species": labels}).to_csv(args.out, index=False)
    np.savez_compressed(os.path.splitext(args.out)[0] + "_probs.npz", treeID=np.array(ids), probs=probs,
                        species=np.array(SPECIES))
    print(f"wrote {args.out} ({len(ids)} trees)")


if __name__ == "__main__":
    main()
