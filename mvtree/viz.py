"""Human-readable previews of the geometric renders."""

from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image, ImageDraw

from .render import VIEW_NAMES


def on_white(img: np.ndarray) -> np.ndarray:
    """Composite a sparse render onto white so it is visible to a person.

    Empty pixels become white; occupied pixels keep their channel values after a
    per-image contrast stretch. Display only: the model consumes raw renders.
    """
    occupied = img.any(axis=-1)
    out = np.full_like(img, 255)
    if occupied.any():
        vals = img[occupied].astype(np.float32)
        lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
        stretched = np.clip((vals - lo) / max(hi - lo, 1.0), 0, 1)
        out[occupied] = (40 + 200 * stretched).astype(np.uint8)
    return out


def montage(views: Dict[str, np.ndarray], tile: int = 192, pad: int = 6, labels: bool = True) -> Image.Image:
    """Seven views side by side, in the canonical order."""
    names = [n for n in VIEW_NAMES if n in views]
    label_h = 16 if labels else 0
    canvas = Image.new("RGB", (len(names) * (tile + pad) + pad, tile + 2 * pad + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(names):
        im = Image.fromarray(on_white(views[name])).resize((tile, tile), Image.NEAREST)
        x = pad + i * (tile + pad)
        canvas.paste(im, (x, pad))
        if labels:
            draw.text((x + 2, pad + tile + 2), name, fill=(60, 60, 60))
    return canvas
