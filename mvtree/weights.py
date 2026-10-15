"""Locate, download, and load released model weights.

Layout under the weights root (default ``./weights``, or ``$MVTREE_WEIGHTS``)::

    weights/
      manifest.json    Google Drive link + SHA-256 of every released file
      convnextv2_256/  config.json  fold0.safetensors ... fold4.safetensors
      convnextv2_512/  config.json  fold0.safetensors ... fold4.safetensors
      geometry_lgbm/   config.json  fold0.txt ... fold4.txt          (optional)

``config.json`` records the backbone, render resolution, normalisation
statistics, class order, and per-fold validation scores.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from .constants import SPECIES
from .model import MultiViewTreeClassifier

MODEL_DIRS = {256: "convnextv2_256", 512: "convnextv2_512"}
MANIFEST = "manifest.json"


def weights_root(root: Optional[str] = None) -> str:
    if root:
        return root
    if os.environ.get("MVTREE_WEIGHTS"):
        return os.environ["MVTREE_WEIGHTS"]
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "weights")


@dataclass
class ModelSpec:
    """One trained resolution: its config and the fold files present on disk."""

    name: str
    path: str
    config: dict
    fold_files: Dict[int, str] = field(default_factory=dict)

    @property
    def img_size(self) -> int:
        return int(self.config["img_size"])

    @property
    def folds(self) -> List[int]:
        return sorted(self.fold_files)


def find_models(root: Optional[str] = None) -> Dict[int, ModelSpec]:
    """Return {img_size: ModelSpec} for every resolution with weights on disk."""
    root = weights_root(root)
    found = {}
    for size, name in MODEL_DIRS.items():
        path = os.path.join(root, name)
        cfg_path = os.path.join(path, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as fh:
            cfg = json.load(fh)
        files = {k: os.path.join(path, f"fold{k}.safetensors") for k in range(5)}
        files = {k: p for k, p in files.items() if os.path.isfile(p)}
        if files:
            found[size] = ModelSpec(name=name, path=path, config=cfg, fold_files=files)
    return found


def load_fold(spec: ModelSpec, fold: int, device: str = "cpu") -> MultiViewTreeClassifier:
    """Build the network and load one fold's weights (stored fp16, run in fp32)."""
    from safetensors.torch import load_file

    cfg = spec.config
    if cfg.get("species", SPECIES) != SPECIES:
        raise ValueError(f"{spec.name}: class order differs from mvtree.constants.SPECIES")
    model = MultiViewTreeClassifier(
        num_classes=len(SPECIES), backbone=cfg["backbone"], pretrained=False,
        img_size=cfg.get("backbone_img_size"),
    )
    state = load_file(spec.fold_files[fold], device="cpu")
    model.load_state_dict({k: v.float() for k, v in state.items()})
    return model.eval().to(device)


def sha256(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(root: Optional[str] = None) -> dict:
    path = os.path.join(weights_root(root), MANIFEST)
    if not os.path.isfile(path):
        return {}
    with open(path) as fh:
        return json.load(fh)


def verify(root: Optional[str] = None, quiet: bool = False) -> bool:
    """Check every file listed in the manifest that is present on disk."""
    root = weights_root(root)
    ok = True
    for rel, meta in read_manifest(root).get("files", {}).items():
        path = os.path.join(root, rel)
        if not os.path.isfile(path):
            continue
        good = sha256(path) == meta["sha256"]
        ok &= good
        if not quiet:
            print(f"{'ok ' if good else 'BAD'} {rel}")
    return ok


def download(url: Optional[str] = None, root: Optional[str] = None) -> str:
    """Download the released weights folder from Google Drive and verify it.

    ``url`` defaults to ``$MVTREE_GDRIVE_URL`` or the ``gdrive_folder`` entry in
    ``weights/manifest.json``. Only files listed in the manifest are fetched, so
    other files in the Drive folder (an old manifest, a README) never overwrite
    the repository's copies, and files already on disk with the right checksum
    are skipped. Needs ``gdown`` (installed with the package).
    """
    import gdown

    root = weights_root(root)
    wanted = read_manifest(root).get("files", {})
    url = url or os.environ.get("MVTREE_GDRIVE_URL") or read_manifest(root).get("gdrive_folder")
    if not url:
        raise RuntimeError("no download link configured: set MVTREE_GDRIVE_URL or 'gdrive_folder' in "
                           f"{os.path.join(root, MANIFEST)}")
    listing = gdown.download_folder(url=url, output=root, quiet=True, skip_download=True)
    for item in listing:
        rel = item.path.replace(os.sep, "/")
        if wanted and rel not in wanted:
            continue
        path = os.path.join(root, rel)
        if rel in wanted and os.path.isfile(path) and sha256(path) == wanted[rel]["sha256"]:
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        gdown.download(id=item.id, output=path, quiet=False)
    missing = [rel for rel in wanted if not os.path.isfile(os.path.join(root, rel))]
    if missing:
        print(f"not in the Drive folder: {', '.join(missing)}")
    if not verify(root):
        raise RuntimeError("checksum mismatch after download; delete the bad files and retry")
    return root


def describe(root: Optional[str] = None) -> str:
    models = find_models(root)
    if not models:
        return f"no weights found under {weights_root(root)}"
    lines = []
    for size, spec in sorted(models.items()):
        lines.append(f"{spec.name}: {size}px, folds {spec.folds}, backbone {spec.config['backbone']}")
    return "\n".join(lines)


def torch_device(requested: Optional[str] = None) -> str:
    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"
