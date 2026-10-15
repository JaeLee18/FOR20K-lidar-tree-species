#!/usr/bin/env python3
"""Convert training checkpoints into the release format.

Training writes ``best_fold{k}.pt`` (model + optimizer + scheduler, ~3.2 GB).
This keeps only the model weights, stores them as fp16 safetensors (~0.53 GB
per fold), and writes ``config.json`` with the metadata the predictor needs.

    python scripts/export/export_weights.py --ckpt-dir runs/convnextv2_256 --img-size 256 \
        --out weights/convnextv2_256
"""

import argparse
import json
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.constants import SPECIES, render_stats  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", required=True, help="directory with best_fold{k}.pt")
    ap.add_argument("--img-size", type=int, required=True, help="render resolution the model was trained on")
    ap.add_argument("--out", required=True, help="output directory, e.g. weights/convnextv2_256")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--backbone-img-size", type=int, default=None,
                    help="fixed input size for ViT backbones (e.g. 224 for DINOv2)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    folds, backbone = {}, None
    for k in range(5):
        src = os.path.join(args.ckpt_dir, f"best_fold{k}.pt")
        if not os.path.isfile(src):
            print(f"fold {k}: {src} missing, skipped")
            continue
        ckpt = torch.load(src, map_location="cpu", weights_only=False)
        classes = list(ckpt["label_encoder"].classes_) if "label_encoder" in ckpt else SPECIES
        if classes != SPECIES:
            raise ValueError(f"{src}: class order differs from mvtree.constants.SPECIES")
        backbone = ckpt.get("backbone", backbone)
        state = {k2: v.to(dtype).contiguous() for k2, v in ckpt["model_state_dict"].items()}
        save_file(state, os.path.join(args.out, f"fold{k}.safetensors"),
                  metadata={"fold": str(k), "epoch": str(ckpt.get("epoch"))})
        folds[k] = {"epoch": ckpt.get("epoch"), "val_oa": ckpt.get("val_oa"),
                    "val_macro_f1": ckpt.get("val_macro_f1")}
        print(f"fold {k}: epoch {ckpt.get('epoch')}, val OA {ckpt.get('val_oa'):.4f}, "
              f"macro-F1 {ckpt.get('val_macro_f1'):.4f} -> fold{k}.safetensors")
        del ckpt, state

    if not folds:
        sys.exit("no checkpoints found")
    config = {
        "architecture": "MultiViewTreeClassifier",
        "backbone": backbone,
        "img_size": args.img_size,
        "backbone_img_size": args.backbone_img_size,
        "render_stats": render_stats(args.img_size),
        "species": SPECIES,
        "dtype": args.dtype,
        "split": "splits/random_5fold.csv",
        "folds": folds,
    }
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)
    print(f"wrote {args.out}/config.json")


if __name__ == "__main__":
    main()
