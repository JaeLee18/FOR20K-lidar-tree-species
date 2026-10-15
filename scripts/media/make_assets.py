#!/usr/bin/env python3
"""Regenerate the documentation figures and the demo animation in ``docs/assets/``.

    python scripts/media/make_assets.py                       # everything
    python scripts/media/make_assets.py --skip-demo           # figures only
    python scripts/media/make_assets.py --paper-figures DIR   # rasterize paper PDFs from DIR

Paper figures are rasterized from the LaTeX sources' PDFs when the directory
exists (default ``../tree_cls/wacv_paper/figures``) and skipped otherwise.
The renders figure and the demo are built from ``examples/*.laz``
the demo
runs the 256 px model (5 folds, TTA 20), so weights must be installed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)
ASSETS = os.path.join(REPO, "docs", "assets")
EXAMPLES = os.path.join(REPO, "examples")

PAPER_FIGURES = {
    "pipeline.png": "fig_method.pdf",
    "species_gallery.png": "fig_species_gallery.pdf",
    "cross_source.png": "fig_lso.pdf",
    "protocol.png": "fig_protocol.pdf",
    "composition.png": "fig_composition.pdf",
    "confusion.png": "fig_confusion.pdf",
    "balanced_metric.png": "fig_balanced_metric.pdf",
}

# Figures that are shown full-width in the README get a higher resolution.
HIGH_DPI = {"pipeline.png": 300, "composition.png": 250}

# (file, true species, platform) for the demo, in display order.
DEMO_TREES = [
    ("00415_Picea_abies_TLS.laz", "Picea_abies", "TLS"),
    ("00883_Fagus_sylvatica_TLS.laz", "Fagus_sylvatica", "TLS"),
    ("03450_Quercus_robur_MLS.laz", "Quercus_robur", "MLS"),
    ("05554_Pinus_sylvestris_ULS.laz", "Pinus_sylvestris", "ULS"),
]

MAX_BYTES = 3 * 1024 * 1024


# ---------------------------------------------------------------------------
# Paper figures
# ---------------------------------------------------------------------------

def _trim_white(img: Image.Image, pad: int = 12, thresh: int = 250) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = (arr < thresh).any(axis=-1)
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    top, bottom = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, arr.shape[0])
    left, right = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, arr.shape[1])
    return img.crop((left, top, right, bottom))


def _save_png(img: Image.Image, path: str) -> None:
    img.save(path, optimize=True)
    if os.path.getsize(path) > MAX_BYTES:  # fall back to a 256-color palette
        img.convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)


def rasterize_paper_figures(fig_dir: str, dpi: int = 200) -> None:
    if not os.path.isdir(fig_dir):
        print(f"[paper figures] {fig_dir} not found, skipped")
        return
    if shutil.which("pdftoppm") is None:
        print("[paper figures] pdftoppm (poppler-utils) not installed, skipped")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for out_name, pdf_name in PAPER_FIGURES.items():
            pdf = os.path.join(fig_dir, pdf_name)
            if not os.path.isfile(pdf):
                print(f"[paper figures] {pdf_name} missing, skipped")
                continue
            stem = os.path.join(tmp, os.path.splitext(out_name)[0])
            r = HIGH_DPI.get(out_name, dpi)
            subprocess.run(["pdftoppm", "-png", "-r", str(r), "-singlefile", pdf, stem], check=True)
            img = Image.open(stem + ".png")
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGBA")
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img)
            img = _trim_white(img.convert("RGB"))
            out = os.path.join(ASSETS, out_name)
            _save_png(img, out)
            print(f"[paper figures] {out_name}: {img.size[0]}x{img.size[1]}, {os.path.getsize(out) / 1e3:.0f} kB")


# ---------------------------------------------------------------------------
# Render explainer
# ---------------------------------------------------------------------------

def white_composite(img: np.ndarray) -> np.ndarray:
    """Show a sparse render on white: empty pixels white, occupied pixels in
    their (contrast-stretched) channel colors, blended toward white by strength."""
    rgb = img.astype(np.float32) / 255.0
    occupied = img.any(axis=-1)
    out = np.ones_like(rgb)
    if occupied.any():
        vals = rgb[occupied]
        lo = np.percentile(vals, 1, axis=0)
        hi = np.percentile(vals, 99, axis=0)
        stretched = np.clip((vals - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
        # Keep colors saturated but never white-on-white: map into [0.15, 0.95].
        color = 0.15 + 0.8 * stretched
        out[occupied] = color * 0.9
    return (out * 255).astype(np.uint8)


def renders_explained(example: str = "00415_Picea_abies_TLS.laz", size: int = 256) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mvtree.io import read_point_cloud
    from mvtree.render import VIEW_NAMES, render_views

    xyz = read_point_cloud(os.path.join(EXAMPLES, example)).astype(np.float64)
    views = render_views(xyz, img_size=size)
    labels = {"side_000": "side 0°", "side_090": "side 90°", "side_180": "side 180°",
              "side_270": "side 270°", "top": "top", "bottom": "bottom", "detail": "trunk detail"}

    fig = plt.figure(figsize=(14, 4.9), facecolor="white")
    gs = fig.add_gridspec(2, 7, height_ratios=[1, 1], hspace=0.28, wspace=0.06)
    for i, name in enumerate(VIEW_NAMES):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(white_composite(views[name]), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
            s.set_linewidth(0.6)
        ax.set_xlabel(labels[name], fontsize=10)

    side = views["side_000"]
    occupied = side.any(axis=-1)
    channels = [("depth", 0, "viridis"), ("log density", 1, "magma"), ("height", 2, "cividis")]
    for j, (title, c, cmap) in enumerate(channels):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2] if j < 3 else gs[1, j])
        chan = np.ma.masked_where(~occupied, side[..., c].astype(float))
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("white")
        ax.imshow(chan, cmap=cm, interpolation="nearest", vmin=0, vmax=255)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#cccccc")
            s.set_linewidth(0.6)
        ax.set_xlabel(f"side 0°: {title} channel", fontsize=10)
    ax = fig.add_subplot(gs[1, 6])
    ax.axis("off")
    ax.text(0.0, 0.5, "R = depth\nG = log(1 + count)\nB = height\n\nempty pixels shown white",
            fontsize=10, va="center", ha="left", color="#333333", family="monospace")

    out = os.path.join(ASSETS, "renders_explained.png")
    fig.savefig(out, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    _save_png(Image.open(out).convert("RGB"), out)
    print(f"[renders] renders_explained.png: {os.path.getsize(out) / 1e3:.0f} kB")


# ---------------------------------------------------------------------------
# Demo animation
# ---------------------------------------------------------------------------

def _pretty(species: str) -> str:
    return species.replace("_", " ")


def demo(fps: int = 15, seconds_per_tree: float = 3.0, max_points: int = 25000,
         width_px: int = 800) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mvtree.io import read_point_cloud
    from mvtree.predict import Predictor

    if shutil.which("ffmpeg") is None:
        print("[demo] ffmpeg not installed, skipped")
        return

    predictor = Predictor(resolutions=[256], tta=20)
    print(f"[demo] predictor: {predictor.describe()}")
    trees = []
    for fname, truth, platform in DEMO_TREES:
        path = os.path.join(EXAMPLES, fname)
        if not os.path.isfile(path):
            print(f"[demo] {fname} missing, skipped")
            continue
        xyz = read_point_cloud(path)
        pred = predictor.predict(xyz)
        top1, conf = pred.top(1)[0]
        print(f"[demo] {fname}: top-1 {top1} ({conf:.3f}), truth {truth}")
        rng = np.random.default_rng(0)
        idx = rng.choice(len(xyz), size=min(max_points, len(xyz)), replace=False)
        pts = xyz[idx].astype(np.float64)
        pts[:, 0] -= np.median(pts[:, 0])
        pts[:, 1] -= np.median(pts[:, 1])
        pts[:, 2] -= pts[:, 2].min()
        trees.append(dict(pts=pts, n=len(xyz), truth=truth, platform=platform, top=pred.top(5)))
    del predictor

    n_frames = int(round(fps * seconds_per_tree))
    dpi = 100
    figsize = (width_px / dpi, width_px * 0.5 / dpi)
    cmap = plt.get_cmap("viridis")
    accent, base = "#d1495b", "#8da9c4"

    with tempfile.TemporaryDirectory() as tmp:
        k = 0
        for t in trees:
            pts = t["pts"]
            h = pts[:, 2].max()
            r = np.abs(pts[:, :2]).max()
            colors = cmap(pts[:, 2] / max(h, 1e-6))
            names = [_pretty(s) for s, _ in t["top"]][::-1]
            probs = [p for _, p in t["top"]][::-1]
            bar_colors = [accent if s == t["truth"] else base for s, _ in t["top"]][::-1]
            for f in range(n_frames):
                fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
                ax3 = fig.add_axes([0.0, 0.0, 0.44, 1.0], projection="3d")
                ax3.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=colors, s=0.4, linewidths=0,
                            depthshade=False)
                ax3.set_xlim(-r, r)
                ax3.set_ylim(-r, r)
                ax3.set_zlim(0, h)
                ax3.set_box_aspect((2 * r, 2 * r, h), zoom=1.05)
                ax3.view_init(elev=12, azim=360.0 * f / n_frames)
                ax3.set_axis_off()
                ax3.set_facecolor("white")

                axb = fig.add_axes([0.62, 0.22, 0.34, 0.6])
                axb.barh(range(len(probs)), probs, color=bar_colors, height=0.65)
                axb.set_yticks(range(len(probs)))
                axb.set_yticklabels(names, fontsize=10, style="italic")
                axb.set_xlim(0, 1)
                axb.set_xticks([0, 0.5, 1.0])
                axb.tick_params(axis="x", labelsize=9)
                axb.set_xlabel("probability", fontsize=9)
                for s in ("top", "right"):
                    axb.spines[s].set_visible(False)
                for i, p in enumerate(probs):
                    axb.text(min(p + 0.02, 0.86), i, f"{p:.2f}", va="center", fontsize=9, color="#333333")
                fig.text(0.62, 0.9, f"{t['platform']}  ·  {t['n']:,} points  ·  height {h:.1f} m",
                         fontsize=10, color="#333333")
                fig.text(0.62, 0.08, f"true: {_pretty(t['truth'])}", fontsize=10, color=accent,
                         style="italic")
                fig.savefig(os.path.join(tmp, f"f{k:04d}.png"), dpi=dpi, facecolor="white")
                plt.close(fig)
                k += 1
        print(f"[demo] {k} frames rendered")

        mp4 = os.path.join(ASSETS, "demo.mp4")
        gif = os.path.join(ASSETS, "demo.gif")
        pattern = os.path.join(tmp, "f%04d.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", mp4], check=True)
        palette = os.path.join(tmp, "palette.png")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                        "-vf", "palettegen=max_colors=128:stats_mode=diff", palette], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                        "-i", palette, "-lavfi", "paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                        "-loop", "0", gif], check=True)
        print(f"[demo] demo.mp4 {os.path.getsize(mp4) / 1e6:.1f} MB, demo.gif {os.path.getsize(gif) / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-figures", default=os.path.join(REPO, "..", "tree_cls", "wacv_paper", "figures"))
    ap.add_argument("--skip-paper", action="store_true")
    ap.add_argument("--skip-renders", action="store_true")
    ap.add_argument("--skip-demo", action="store_true")
    args = ap.parse_args()
    os.makedirs(ASSETS, exist_ok=True)
    if not args.skip_paper:
        rasterize_paper_figures(os.path.abspath(args.paper_figures))
    if not args.skip_renders:
        renders_explained()
    if not args.skip_demo:
        demo()


if __name__ == "__main__":
    main()
