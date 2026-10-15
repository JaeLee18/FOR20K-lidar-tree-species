"""Command-line interface: ``mvtree {predict,render,download,serve,info}``."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time


def _cmd_predict(args: argparse.Namespace) -> int:
    from tqdm import tqdm

    from .io import find_point_clouds
    from .predict import Predictor
    from .render import save_views
    from .viz import montage

    paths = find_point_clouds(args.inputs)
    if not paths:
        print("no point clouds found", file=sys.stderr)
        return 1
    folds = [int(f) for f in args.folds.split(",")] if args.folds else None
    resolutions = None if args.resolution == "auto" else [256, 512] if args.resolution == "multi" \
        else [int(args.resolution)]
    predictor = Predictor(weights=args.weights, resolutions=resolutions, folds=folds, device=args.device,
                          tta=args.tta, seed=args.seed, low_memory=args.low_memory, max_batch=args.max_batch)
    print(f"[mvtree] {len(paths)} scan(s) | model: {predictor.describe()}", file=sys.stderr)
    n_passes = sum(len(f) for f in predictor.folds.values()) * predictor.tta
    if predictor.device == "cpu" and n_passes > 10:
        print(f"[mvtree] running {n_passes} forward passes per tree on CPU (a few minutes each); "
              "use --folds 0 --tta 1 for a fast preview.", file=sys.stderr)
    if args.resolution == "auto" and predictor.resolutions == [256]:
        print("[mvtree] note: only 256 px weights are installed; the paper's best model also "
              "averages the 512 px model (see weights/README.md).", file=sys.stderr)

    results, t0 = [], time.time()
    keep_views = bool(args.save_renders)
    for pred in tqdm(predictor.predict_files(paths, keep_views=keep_views), total=len(paths),
                     disable=len(paths) == 1 or args.quiet, file=sys.stderr):
        results.append(pred)
        if args.save_renders:
            stem = os.path.splitext(os.path.basename(pred.path))[0]
            for res, views in pred.views.items():
                save_views(views, os.path.join(args.save_renders, stem, str(res)))
                montage(views).save(os.path.join(args.save_renders, stem, f"montage_{res}.png"))
    elapsed = time.time() - t0

    if not args.quiet:
        for r in results:
            print(f"\n{os.path.basename(r.path)}  ({r.n_points:,} points, height {r.tree_height:.1f} m)")
            for rank, (sp, p) in enumerate(r.top(args.topk), 1):
                bar = "#" * int(round(p * 30))
                print(f"  {rank}. {sp:<24s} {p:6.1%}  {bar}")
        print(f"\n[mvtree] {len(results)} scan(s) in {elapsed:.1f}s", file=sys.stderr)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        if args.output.endswith(".json"):
            with open(args.output, "w") as fh:
                json.dump([r.to_dict(args.topk, with_probs=True) for r in results], fh, indent=2)
        else:
            with open(args.output, "w", newline="") as fh:
                w = csv.writer(fh)
                header = ["file", "n_points", "tree_height_m", "species", "genus", "confidence"]
                for k in range(2, args.topk + 1):
                    header += [f"top{k}_species", f"top{k}_prob"]
                w.writerow(header)
                for r in results:
                    row = [r.path, r.n_points, f"{r.tree_height:.3f}", r.species, r.genus, f"{r.confidence:.4f}"]
                    for sp, p in r.top(args.topk)[1:]:
                        row += [sp, f"{p:.4f}"]
                    w.writerow(row)
        print(f"[mvtree] wrote {args.output}", file=sys.stderr)
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    from .io import find_point_clouds, read_point_cloud
    from .render import render_views, save_views
    from .viz import montage

    for path in find_point_clouds(args.inputs):
        stem = os.path.splitext(os.path.basename(path))[0]
        views = render_views(read_point_cloud(path).astype("float64"), img_size=args.size)
        out = os.path.join(args.output, stem)
        save_views(views, out)
        montage(views).save(os.path.join(args.output, f"{stem}_montage.png"))
        print(f"{path} -> {out}/")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    from .weights import describe, download, verify

    if args.verify_only:
        return 0 if verify(args.weights) else 1
    try:
        root = download(url=args.url, root=args.weights)
    except RuntimeError as exc:
        print(f"[mvtree] {exc}", file=sys.stderr)
        return 1
    print(f"weights in {root}\n{describe(root)}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import torch

    from . import __version__
    from .weights import describe, weights_root

    print(f"mvtree {__version__} | torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"weights root: {weights_root(args.weights)}")
    print(describe(args.weights))
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    # A long-running server fragments the CUDA cache; expandable segments avoid
    # spurious out-of-memory errors (must be set before CUDA initialises).
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from .web.server import serve

    def make_predictor():
        from .predict import Predictor

        resolutions = None if args.resolution == "auto" else [256, 512] if args.resolution == "multi" \
            else [int(args.resolution)]
        folds = [int(f) for f in args.folds.split(",")] if args.folds else None
        return Predictor(weights=args.weights, resolutions=resolutions, folds=folds, device=args.device,
                         tta=args.tta, seed=args.seed, low_memory=args.low_memory, max_batch=args.max_batch)

    token = args.token or os.environ.get("MVTREE_TOKEN") or None
    return serve(args.host, args.port, args.data_dir, make_predictor, keep_scans=not args.no_keep_scans,
                 token=token, max_upload_mb=args.max_upload_mb)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="mvtree", description="Tree-species recognition from single-tree LiDAR scans.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="classify one or more single-tree scans")
    p.add_argument("inputs", nargs="+", help="point-cloud files and/or directories")
    p.add_argument("-o", "--output", help="write results to .csv or .json")
    p.add_argument("-k", "--topk", type=int, default=5)
    p.add_argument("-r", "--resolution", default="auto", choices=["auto", "256", "512", "multi"],
                   help="auto = every installed resolution (default)")
    p.add_argument("--folds", help="comma-separated fold subset, e.g. 0 or 0,1 (default: all)")
    p.add_argument("--tta", type=int, default=20, help="augmented passes per model (paper: 20; 1 = off)")
    p.add_argument("--device", default="auto", help="cuda, cuda:1, cpu, or auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-batch", type=int, default=20, help="augmented copies per forward pass")
    p.add_argument("--low-memory", action="store_true", help="hold one network in memory at a time")
    p.add_argument("--save-renders", metavar="DIR", help="also save the 7 views and a preview montage")
    p.add_argument("--weights", help="weights root (default ./weights or $MVTREE_WEIGHTS)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=_cmd_predict)

    r = sub.add_parser("render", help="render the 7 orthographic views without classifying")
    r.add_argument("inputs", nargs="+")
    r.add_argument("-o", "--output", default="renders")
    r.add_argument("-s", "--size", type=int, default=256)
    r.set_defaults(func=_cmd_render)

    d = sub.add_parser("download", help="download released weights from Google Drive")
    d.add_argument("--url", help="Google Drive folder link (default: weights/manifest.json)")
    d.add_argument("--verify-only", action="store_true", help="only check SHA-256 of files on disk")
    d.add_argument("--weights", help="destination root")
    d.set_defaults(func=_cmd_download)

    s = sub.add_parser("serve", help="self-hosted web app: upload scans in a browser and keep a record of predictions")
    s.add_argument("--host", default="127.0.0.1", help="interface to listen on (0.0.0.0 = every network interface)")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--data-dir", default="mvtree_data", help="where records, scans, and previews are saved")
    s.add_argument("--token", help="require this access token (or set $MVTREE_TOKEN); use it whenever --host "
                                   "is not 127.0.0.1")
    s.add_argument("--no-keep-scans", action="store_true",
                   help="delete each uploaded scan after prediction (records keep results and previews)")
    s.add_argument("--max-upload-mb", type=float, default=500)
    s.add_argument("-r", "--resolution", default="auto", choices=["auto", "256", "512", "multi"])
    s.add_argument("--folds", help="comma-separated fold subset, e.g. 0 for a fast CPU server")
    s.add_argument("--tta", type=int, default=20, help="augmented passes per model (paper: 20; 1 = off)")
    s.add_argument("--device", default="auto", help="cuda, cuda:1, cpu, or auto")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-batch", type=int, default=20)
    s.add_argument("--low-memory", action="store_true", help="hold one network in memory at a time")
    s.add_argument("--weights", help="weights root (default ./weights or $MVTREE_WEIGHTS)")
    s.set_defaults(func=_cmd_serve)

    i = sub.add_parser("info", help="show environment and installed weights")
    i.add_argument("--weights")
    i.set_defaults(func=_cmd_info)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
