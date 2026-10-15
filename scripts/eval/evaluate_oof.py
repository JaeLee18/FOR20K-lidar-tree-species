#!/usr/bin/env python3
"""Out-of-fold metrics on the public labeled set (paper Table 2 and Sec. 3).

Reports natural-distribution overall accuracy (OA), macro-F1, balanced accuracy,
and per-LiDAR-type OA. Because the official hidden test is species-balanced,
its OA should be compared with the macro metrics here, not with OA.

    # the ablation rows shipped in results/oof
    python scripts/eval/evaluate_oof.py results/oof/densenet201_256_random.npz \
        results/oof/dinov2_224_random.npz results/oof/convnextv2_256_random.npz

    # your own runs; --ensemble adds the averaged multi-resolution row
    python scripts/eval/evaluate_oof.py runs/convnextv2_256 runs/convnextv2_512 --ensemble
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.evaluation import attach_metadata, average_oof, load_oof, per_species_f1, summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="OOF .npz files or run directories with oof_fold*.npz")
    ap.add_argument("--metadata", default="data/metadata/tree_metadata_dev.csv")
    ap.add_argument("--ensemble", action="store_true", help="also report the probability average of all runs")
    ap.add_argument("--per-species", action="store_true", help="print per-species F1 for the last row")
    args = ap.parse_args()

    meta = pd.read_csv(args.metadata)
    frames = {os.path.basename(r.rstrip("/")).replace(".npz", ""): load_oof(r) for r in args.runs}
    if args.ensemble and len(frames) > 1:
        frames["ensemble(" + "+".join(frames) + ")"] = average_oof(list(frames.values()))

    rows = {name: summary(attach_metadata(df, meta)) for name, df in frames.items()}
    table = pd.DataFrame(rows).T
    table["n"] = table["n"].astype(int)
    table["OA-macroF1"] = table["OA"] - table["macro_F1"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200,
                           "display.max_columns", None):
        print(table)
    if args.per_species:
        last = attach_metadata(list(frames.values())[-1], meta)
        print("\nper-species F1 (" + list(frames)[-1] + ")")
        print(per_species_f1(last).sort_values().to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
