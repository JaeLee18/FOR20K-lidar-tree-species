#!/usr/bin/env python3
"""Export one fold as a compact ONNX model for the offline Android app.

    python scripts/export/export_mobile.py --img-size 256 --fold 0 --per-channel --out android/app/src/main/assets \
        --eval-renders data/renders_256/dev

The full model (5 folds x 2 resolutions, 5.3 GB) does not fit on a phone, so the
app runs one fold at one resolution, without test-time augmentation:

1. load the fold in fp32 and export it to ONNX with batch size 1; the graph ends
   in a softmax, so the app reads class probabilities directly;
2. quantise every MatMul to int8 (dynamic quantisation). The ConvNeXt 1x1
   projections are MatMuls and hold almost all the weights, so the file shrinks
   about 3.6x;
3. optionally score the fold's held-out trees with PyTorch fp32, ONNX fp32, and
   ONNX int8 (from the pre-rendered views), so the accuracy cost of each step
   is measured rather than assumed.

Writes ``model.onnx`` (int8) and ``model_info.json`` (class order, render
statistics, and the measured accuracy) into ``--out``.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.constants import SPECIES, render_stats  # noqa: E402
from mvtree.weights import find_models, load_fold  # noqa: E402

INPUTS = ["side_views", "topbot_views", "detail_view", "tree_height"]


class ProbabilityModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, side_views, topbot_views, detail_view, tree_height):
        return self.model(side_views, topbot_views, detail_view, tree_height).softmax(dim=-1)


def dummy_inputs(size: int):
    g = torch.Generator().manual_seed(0)
    return (torch.randn(1, 4, 3, size, size, generator=g), torch.randn(1, 2, 3, size, size, generator=g),
            torch.randn(1, 3, size, size, generator=g), torch.tensor([[15.0]]))


def export_onnx(model: torch.nn.Module, size: int, path: str, opset: int) -> None:
    args = dummy_inputs(size)
    try:
        torch.onnx.export(model, args, path, input_names=INPUTS, output_names=["probs"], opset_version=opset,
                          dynamo=True, external_data=False)
    except Exception as exc:  # fall back to the TorchScript exporter
        print(f"dynamo export failed ({type(exc).__name__}: {exc}); using the TorchScript exporter")
        torch.onnx.export(model, args, path, input_names=INPUTS, output_names=["probs"], opset_version=opset,
                          dynamo=False)


def quantize(fp32_path: str, int8_path: str, per_channel: bool = False) -> None:
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    # The exporter's intermediate shape annotations trip the quantiser's shape inference
    # (it re-infers them anyway), so drop them first.
    model = onnx.load(fp32_path)
    model.graph.ClearField("value_info")
    onnx.save(model, fp32_path)
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"],
                     per_channel=per_channel)


def session(path: str, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    from sklearn.metrics import f1_score

    pred = probs.argmax(1)
    return {"oa": round(float((pred == labels).mean()), 4),
            "macro_f1": round(float(f1_score(labels, pred, average="macro")), 4), "n": int(len(labels))}


def evaluate(model, sessions: dict, args) -> dict:
    """Score PyTorch and each ONNX model on the fold's held-out trees (no TTA)."""
    import pandas as pd

    from mvtree.data import TreeRenderDataset, has_renders

    meta = pd.read_csv(args.metadata)
    split = pd.read_csv(args.split)
    val_ids = set(split.treeID[split[f"fold_{args.fold}"] == "val"])
    meta = meta[meta.treeID.isin(val_ids)]
    meta = meta[[has_renders(args.eval_renders, t) for t in meta.treeID]].sort_values("treeID")
    if args.eval_limit:
        meta = meta.sample(n=min(args.eval_limit, len(meta)), random_state=0).sort_values("treeID")
    labels = meta.species.map({s: i for i, s in enumerate(SPECIES)}).values
    ds = TreeRenderDataset(meta.treeID.values, labels, meta.tree_H.values, args.eval_renders,
                           img_size=args.img_size, augment=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    out = {name: [] for name in ["pytorch_fp32", *sessions]}
    t0 = time.time()
    with torch.no_grad():
        for i, (side, topbot, detail, height, _) in enumerate(loader):
            out["pytorch_fp32"].append(model(side.to(device), topbot.to(device), detail.to(device),
                                             height.to(device)).float().cpu().numpy()[0])
            feeds = dict(zip(INPUTS, (side.numpy(), topbot.numpy(), detail.numpy(), height.numpy())))
            for name, sess in sessions.items():
                out[name].append(sess.run(None, feeds)[0][0])
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(ds)} trees, {(time.time() - t0) / (i + 1):.2f} s/tree", flush=True)
    model.cpu()
    ref = np.asarray(out["pytorch_fp32"])
    report = {}
    for name, probs in out.items():
        probs = np.asarray(probs)
        report[name] = metrics(probs, labels)
        if name != "pytorch_fp32":
            report[name]["top1_agreement_with_pytorch"] = round(float((probs.argmax(1) == ref.argmax(1)).mean()), 4)
            report[name]["max_abs_prob_diff"] = round(float(np.abs(probs - ref).max()), 4)
    os.makedirs(os.path.dirname(os.path.abspath(args.eval_out)), exist_ok=True)
    np.savez_compressed(args.eval_out, treeID=meta.treeID.values, labels=labels,
                        **{k: np.asarray(v, dtype=np.float32) for k, v in out.items()})
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="android/app/src/main/assets")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--eval-renders", help="pre-rendered views of the dev set, e.g. data/renders_256/dev")
    ap.add_argument("--eval-limit", type=int, default=0, help="score a random subset (0 = all held-out trees)")
    ap.add_argument("--eval-fp32-onnx", action="store_true", help="also score the unquantised ONNX model")
    ap.add_argument("--split", default="splits/random_5fold.csv")
    ap.add_argument("--metadata", default="data/metadata/tree_metadata_dev.csv")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--per-channel", action="store_true", help="one int8 scale per output channel")
    ap.add_argument("--eval-out", default="runs/mobile/eval_probs.npz", help="per-tree probabilities of the check")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spec = find_models(args.weights).get(args.img_size)
    if spec is None or args.fold not in spec.fold_files:
        sys.exit(f"no {args.img_size}px fold {args.fold} weights installed")
    model = ProbabilityModel(load_fold(spec, args.fold, "cpu")).eval()

    fp32_path = os.path.join(args.out, "model_fp32.onnx")
    int8_path = os.path.join(args.out, "model.onnx")
    export_onnx(model, args.img_size, fp32_path, args.opset)
    quantize(fp32_path, int8_path, per_channel=args.per_channel)
    print(f"fp32 {os.path.getsize(fp32_path) / 2**20:.0f} MB -> int8 {os.path.getsize(int8_path) / 2**20:.0f} MB")

    # Sanity check on random input before the (slow) accuracy run.
    feeds = dict(zip(INPUTS, (t.numpy() for t in dummy_inputs(args.img_size))))
    with torch.no_grad():
        ref = model(*dummy_inputs(args.img_size)).numpy()
    for name, path in [("fp32", fp32_path), ("int8", int8_path)]:
        got = session(path, args.threads).run(None, feeds)[0]
        print(f"ONNX {name} vs PyTorch on random input: max |dp| = {np.abs(got - ref).max():.2e}")

    info = {
        "format": 1,
        "architecture": "MultiViewTreeClassifier (ConvNeXtV2-Base x 3)",
        "backbone": spec.config.get("backbone"),
        "img_size": args.img_size,
        "fold": args.fold,
        "tta": 1,
        "quantization": f"dynamic int8 (MatMul/Gemm weights, {'per-channel' if args.per_channel else 'per-tensor'}),"
                        " fp32 activations",
        "inputs": {"side_views": [1, 4, 3, args.img_size, args.img_size],
                   "topbot_views": [1, 2, 3, args.img_size, args.img_size],
                   "detail_view": [1, 3, args.img_size, args.img_size], "tree_height": [1, 1]},
        "output": "probs [1, 33], softmax in the order of `species`",
        "species": SPECIES,
        "render_stats": render_stats(args.img_size),
        "sha256": hashlib.sha256(open(int8_path, "rb").read()).hexdigest(),
    }
    if args.eval_renders:
        sessions = {"onnx_int8": session(int8_path, args.threads)}
        if args.eval_fp32_onnx:
            sessions["onnx_fp32"] = session(fp32_path, args.threads)
        info["evaluation"] = {"trees": f"held-out trees of fold {args.fold}", "tta": 1,
                              **evaluate(model, sessions, args)}
        print(json.dumps(info["evaluation"], indent=2))
    with open(os.path.join(args.out, "model_info.json"), "w") as fh:
        json.dump(info, fh, indent=2)
    os.remove(fp32_path)
    print(f"wrote {int8_path} and model_info.json")


if __name__ == "__main__":
    main()
