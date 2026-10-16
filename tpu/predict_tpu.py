#!/usr/bin/env python3
"""Classify single-tree LiDAR scans on a Cloud TPU (PyTorch/XLA).

Runs the same ensemble as ``mvtree predict``: every installed resolution, the
five fold models, and 20 test-time augmentations per model. Each TPU chip on
the host takes every n-th scan, keeps all fold models in its memory, and
classifies several trees per forward pass while CPU worker processes read and
render the next scans. The augmentation draws are those of ``mvtree.predict``,
so the probabilities match the GPU predictor's up to bf16 rounding
(``--precision fp32`` to compare closely).

    python tpu/predict_tpu.py examples/
    python tpu/predict_tpu.py scans/ -o predictions.csv
    python tpu/predict_tpu.py tree.laz --chips 1 --folds 0 --tta 1

Options and output columns follow ``mvtree predict`` (docs/inference.md).
"""

import argparse
import functools
import multiprocessing as mp
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xla_common  # noqa: E402  (puts the repository root on sys.path)

from mvtree.constants import render_stats  # noqa: E402
from mvtree.io import find_point_clouds, read_point_cloud  # noqa: E402
from mvtree.render import VIEW_NAMES, render_views  # noqa: E402

N_VIEWS = len(VIEW_NAMES)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="point-cloud files and/or directories")
    ap.add_argument("-o", "--output", help="write results to .csv or .json")
    ap.add_argument("-k", "--topk", type=int, default=5)
    ap.add_argument("-r", "--resolution", default="auto", choices=["auto", "256", "512", "multi"],
                    help="auto = every installed resolution (default)")
    ap.add_argument("--folds", help="comma-separated fold subset, e.g. 0 or 0,1 (default: all)")
    ap.add_argument("--tta", type=int, default=20, help="augmented passes per model (paper: 20; 1 = off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", help="weights root (default ./weights or $MVTREE_WEIGHTS)")
    ap.add_argument("--chips", default="all", choices=["all", "1"], help="TPU chips to use on this host")
    ap.add_argument("--trees-per-batch", type=int, default=4,
                    help="trees per forward pass at 256 px, each with --tta copies; a quarter as many at 512 px")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--render-workers", type=int, default=0, help="CPU processes rendering per chip (0 = auto)")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap.parse_args(argv)


def select_models(args):
    """The fold models to run, chosen exactly as ``mvtree predict`` does (nothing is loaded)."""
    from mvtree.predict import Predictor

    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    resolutions = None if args.resolution == "auto" else [256, 512] if args.resolution == "multi" \
        else [int(args.resolution)]
    return Predictor(weights=args.weights, resolutions=resolutions, folds=folds, device="cpu", tta=args.tta)


def tta_indices(n_runs: int, rng: np.random.Generator) -> np.ndarray:
    """(n_runs, 7) indices into [7 views, 7 flipped views] for the augmented copies.

    Draws from ``rng`` in the order of ``mvtree.predict._tta_batch`` (shift,
    then one flip per side, top/bottom, and detail view), so copy j has the
    same shift and flips as the GPU predictor's copy j.
    """
    idx = np.empty((n_runs, N_VIEWS), dtype=np.int64)
    for j in range(n_runs):
        shift = int(rng.integers(0, 4)) if n_runs > 1 else 0
        flips = [bool(rng.random() < 0.5) if n_runs > 1 else False for _ in range(N_VIEWS)]
        views = [(k + shift) % 4 for k in range(4)] + [4, 5, 6]
        idx[j] = [v + N_VIEWS * int(f) for v, f in zip(views, flips)]
    return idx


def prepare(path: str, resolutions) -> dict:
    """Read and render one scan (runs in a CPU worker process)."""
    try:
        xyz = read_point_cloud(path).astype(np.float64)
        views = {}
        for r in resolutions:
            rendered = render_views(xyz, img_size=r)
            views[r] = np.stack([rendered[name] for name in VIEW_NAMES])  # (7, H, W, 3) uint8
        return {"path": path, "n": len(xyz), "height": float(xyz[:, 2].max() - xyz[:, 2].min()), "views": views}
    except Exception as exc:  # report and keep going
        return {"path": path, "error": f"{type(exc).__name__}: {exc}"}


def classify(batch, models, predictor, args, device):
    """Ensemble probabilities for a list of rendered trees: {resolution: (n, 33)}."""
    import torch

    with torch.no_grad():
        return _classify(batch, models, predictor, args, device)


def _classify(batch, models, predictor, args, device):
    import torch
    import torch_xla

    per_res = {}
    for r in predictor.resolutions:
        size = max(1, round(args.trees_per_batch * (256 / r) ** 2))
        stats = render_stats(r)
        mean = torch.tensor(stats["mean"], device=device).view(1, 1, 3, 1, 1)
        std = torch.tensor(stats["std"], device=device).view(1, 1, 3, 1, 1)
        out = []
        for s in range(0, len(batch), size):
            part = batch[s:s + size]
            part = part + [part[-1]] * (size - len(part))  # fixed shapes: one compilation per resolution
            views = torch.from_numpy(np.stack([it["views"][r] for it in part])).to(device)
            x = (views.permute(0, 1, 4, 2, 3).float().div(255) - mean) / std  # (size, 7, 3, H, W)
            both = torch.cat([x, x.flip(-1)], dim=1)  # (size, 14, 3, H, W)
            heights = np.repeat(np.array([it["height"] for it in part], dtype=np.float32), args.tta)
            h = torch.from_numpy(heights[:, None]).to(device)
            torch_xla.sync()
            probs = []
            for fold in predictor.folds[r]:
                idx = tta_indices(args.tta, np.random.default_rng((args.seed, r, fold)))
                g = both[:, torch.from_numpy(idx).to(device)].flatten(0, 1)  # (size * tta, 7, 3, H, W)
                with xla_common.autocast(args.precision):
                    logits = models[(r, fold)](g[:, :4], g[:, 4:6], g[:, 6], h)
                probs.append(logits.float().softmax(-1).view(size, args.tta, -1).mean(1))
                torch_xla.sync()
            out.append(torch.stack(probs).mean(0))
        per_res[r] = torch.cat(out)[:len(batch)]
    return {r: p.cpu() for r, p in per_res.items()}  # one device-to-host copy per batch


def run_chip(index, args, paths):
    """Classify this chip's share of ``paths``; rank 0 returns everything, other ranks None."""
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr

    from mvtree.weights import load_fold

    world, rank = xr.world_size(), xr.global_ordinal()
    xla_common.share_cpu(world)
    device = torch_xla.device()
    predictor = select_models(args)
    t0 = time.time()
    models = {(r, f): load_fold(predictor.specs[r], f, "cpu").to(device)
              for r in predictor.resolutions for f in predictor.folds[r]}
    t_load = time.time() - t0

    mine = list(range(rank, len(paths), world))
    batches = [mine[i:i + args.trees_per_batch] for i in range(0, len(mine), args.trees_per_batch)]
    workers = args.render_workers or max(1, min(2 * args.trees_per_batch, (os.cpu_count() or 2) // world // 2))
    work = functools.partial(prepare, resolutions=tuple(predictor.resolutions))
    results, errors, t_first = [], [], None
    t0 = time.time()
    # spawn, not fork: this process already runs the TPU runtime
    with mp.get_context("spawn").Pool(workers) as pool:
        pending = deque()

        def submit(i):
            if i < len(batches):
                pending.append((batches[i], pool.map_async(work, [paths[j] for j in batches[i]])))

        submit(0)
        submit(1)
        nxt = 2
        while pending:
            ids, fut = pending.popleft()
            submit(nxt)  # keep two batches rendering while the chip runs this one
            nxt += 1
            ready = []
            for j, it in zip(ids, fut.get()):
                if "error" in it:
                    errors.append((j, it["error"]))
                else:
                    ready.append((j, it))
            if not ready:
                continue
            probs = classify([it for _, it in ready], models, predictor, args, device)
            for i, (j, it) in enumerate(ready):
                results.append((j, it["n"], it["height"], {r: p[i].clone() for r, p in probs.items()}))
            if t_first is None:
                t_first = time.time() - t0
    stats = {"rank": rank, "trees": len(results), "load_s": t_load, "first_batch_s": t_first or 0.0,
             "total_s": time.time() - t0, "compiles": xla_common.compile_count()}
    if world > 1:
        gathered = xm.mesh_reduce("mvtree_predict", (results, errors, stats), lambda parts: parts)
    else:
        gathered = [(results, errors, stats)]
    if rank == 0:
        _report(args, paths, predictor, gathered, world)


def _report(args, paths, predictor, gathered, world):
    from mvtree.cli import print_predictions, write_predictions
    from mvtree.predict import Prediction

    results, errors = {}, {}
    for res, errs, _ in gathered:
        results.update({j: (n, h, pr) for j, n, h, pr in res})
        errors.update(dict(errs))
    preds = []
    for j in sorted(results):
        n, h, per_res = results[j]
        per_res = {r: p.numpy() for r, p in per_res.items()}
        preds.append(Prediction(path=paths[j], n_points=n, tree_height=h,
                                probs=np.mean(list(per_res.values()), axis=0), per_resolution=per_res))
    if not args.quiet:
        print_predictions(preds, args.topk)
    for j in sorted(errors):
        print(f"[tpu] skipped {paths[j]}: {errors[j]}", file=sys.stderr)
    stats = [s for _, _, s in gathered]
    total = max(s["total_s"] for s in stats)
    first = max(s["first_batch_s"] for s in stats)
    print(f"\n[tpu] {len(preds)} scan(s) on {world} chip(s) in {total:.1f}s "
          f"(model load {max(s['load_s'] for s in stats):.0f}s, first batch incl. compilation {first:.0f}s, "
          f"{max(s['compiles'] for s in stats)} compilations per chip)", file=sys.stderr)
    if len(preds) > world * args.trees_per_batch and total > first:
        print(f"[tpu] steady state: {(len(preds) - world * args.trees_per_batch) / (total - first):.2f} trees/s",
              file=sys.stderr)
    if args.output:
        write_predictions(preds, args.output, args.topk)
        print(f"[tpu] wrote {args.output}", file=sys.stderr)


def main(argv=None) -> int:
    args = parse_args(argv)
    xla_common.require_xla()
    paths = find_point_clouds(args.inputs)
    if not paths:
        print("no point clouds found", file=sys.stderr)
        return 1
    predictor = select_models(args)  # fail here, before starting the chips, if weights are missing
    parts = [f"{r}px x {len(predictor.folds[r])} folds" for r in predictor.resolutions]
    print(f"[tpu] {len(paths)} scan(s) | model: {' + '.join(parts)}, TTA {args.tta}, {args.precision}",
          file=sys.stderr)
    xla_common.launch(run_chip, (args, paths), all_chips=args.chips == "all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
