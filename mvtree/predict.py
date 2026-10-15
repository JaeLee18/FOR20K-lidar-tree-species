"""Species prediction for single-tree LiDAR scans.

Pipeline per tree: read points -> render 7 views at every available resolution
-> each fold model runs ``tta`` passes (random horizontal flips and a cyclic
shift of the side views, as in training) -> softmax probabilities are averaged
over passes, then over the five folds of a resolution, then across
resolutions. With both 256 and 512 px weights installed this is the paper's
multi-resolution model; with one resolution it is the single-resolution model.

Example:
    >>> from mvtree.predict import Predictor
    >>> p = Predictor()                      # all weights found under ./weights
    >>> result = p.predict_file("examples/05554_Pinus_sylvestris_ULS.laz")
    >>> result.top(3)
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .constants import GENUS, SPECIES, render_stats
from .io import read_point_cloud
from .render import DETAIL_VIEW, SIDE_VIEWS, TOPBOT_VIEWS, render_views
from .weights import ModelSpec, find_models, load_fold, torch_device, weights_root


@dataclass
class Prediction:
    path: Optional[str]
    n_points: int
    tree_height: float
    probs: np.ndarray
    per_resolution: Dict[int, np.ndarray] = field(default_factory=dict)
    views: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict, repr=False)

    @property
    def species(self) -> str:
        return SPECIES[int(self.probs.argmax())]

    @property
    def genus(self) -> str:
        return GENUS[self.species]

    @property
    def confidence(self) -> float:
        return float(self.probs.max())

    def top(self, k: int = 5) -> List[Tuple[str, float]]:
        idx = np.argsort(-self.probs)[:k]
        return [(SPECIES[i], float(self.probs[i])) for i in idx]

    def to_dict(self, k: int = 5, with_probs: bool = False) -> dict:
        out = {
            "file": self.path,
            "n_points": self.n_points,
            "tree_height_m": round(self.tree_height, 3),
            "species": self.species,
            "genus": self.genus,
            "confidence": round(self.confidence, 4),
            "top_k": [{"species": s, "prob": round(p, 4)} for s, p in self.top(k)],
            "resolutions": sorted(self.per_resolution),
        }
        if with_probs:
            out["probs"] = {s: float(p) for s, p in zip(SPECIES, self.probs)}
            out["per_resolution_probs"] = {
                str(r): {s: float(p) for s, p in zip(SPECIES, pr)} for r, pr in self.per_resolution.items()
            }
        return out


def _to_tensors(views: Dict[str, np.ndarray], img_size: int):
    """uint8 renders -> normalised (image, flipped image) tensors per view."""
    stats = render_stats(img_size)
    mean = torch.tensor(stats["mean"]).view(3, 1, 1)
    std = torch.tensor(stats["std"]).view(3, 1, 1)
    out = {}
    for name, img in views.items():
        t = (torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255) - mean) / std
        out[name] = (t, t.flip(-1))
    return out


def _tta_batch(tensors, n_runs: int, rng: np.random.Generator):
    """Draw ``n_runs`` augmented copies of one tree (same policy as training)."""
    side, topbot, detail = [], [], []
    for _ in range(n_runs):
        shift = int(rng.integers(0, 4)) if n_runs > 1 else 0
        flip = (lambda: bool(rng.random() < 0.5)) if n_runs > 1 else (lambda: False)
        side.append(torch.stack([tensors[SIDE_VIEWS[(j + shift) % 4]][int(flip())] for j in range(4)]))
        topbot.append(torch.stack([tensors[v][int(flip())] for v in TOPBOT_VIEWS]))
        detail.append(tensors[DETAIL_VIEW][int(flip())])
    return torch.stack(side), torch.stack(topbot), torch.stack(detail)


@torch.no_grad()
def tta_probs(model, tensors, height: float, n_runs: int, rng: np.random.Generator,
              device: str, max_batch: int = 20) -> np.ndarray:
    """Average softmax over ``n_runs`` augmented passes (n_runs=1: no augmentation)."""
    side, topbot, detail = _tta_batch(tensors, n_runs, rng)
    probs = []
    for i in range(0, n_runs, max_batch):
        s, t, d = (x[i:i + max_batch].to(device) for x in (side, topbot, detail))
        h = torch.full((s.shape[0], 1), float(height), device=device)
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            logits = model(s, t, d, h)
        probs.append(logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(probs).mean(axis=0)


class Predictor:
    """Five-fold, multi-resolution ensemble with test-time augmentation.

    Args:
        weights: Weights root (default ``./weights`` or ``$MVTREE_WEIGHTS``).
        resolutions: Subset of {256, 512}; default all installed.
        folds: Subset of fold indices; default all installed.
        device: ``"cuda"``, ``"cpu"``, or ``None`` for automatic.
        tta: Augmented passes per fold model (the paper uses 20; 1 disables TTA).
        seed: Seed for the augmentation draws, so outputs are reproducible.
        low_memory: Keep one network in memory at a time instead of all of them.
        max_batch: Augmented copies per forward pass (lower it on small GPUs).
    """

    def __init__(self, weights: Optional[str] = None, resolutions: Optional[Sequence[int]] = None,
                 folds: Optional[Sequence[int]] = None, device: Optional[str] = None,
                 tta: int = 20, seed: int = 0, low_memory: bool = False, max_batch: int = 20) -> None:
        available = find_models(weights)
        if not available:
            raise FileNotFoundError(
                f"no model weights under {weights_root(weights)}. "
                f"Download them with `python -m mvtree.cli download` (see weights/README.md)."
            )
        if resolutions:
            missing = [r for r in resolutions if r not in available]
            if missing:
                raise FileNotFoundError(f"weights for {missing} px not installed; have {sorted(available)}")
            available = {r: available[r] for r in resolutions}
        self.specs: Dict[int, ModelSpec] = available
        self.folds: Dict[int, List[int]] = {}
        for r, spec in self.specs.items():
            chosen = [f for f in spec.folds if folds is None or f in folds]
            if not chosen:
                raise FileNotFoundError(f"none of folds {folds} installed for {r}px (have {spec.folds})")
            self.folds[r] = chosen
        self.device = torch_device(device)
        self.tta = max(1, int(tta))
        self.seed = seed
        self.low_memory = low_memory
        self.max_batch = max_batch
        self._cache: Dict[Tuple[int, int], torch.nn.Module] = {}

    @property
    def resolutions(self) -> List[int]:
        return sorted(self.specs)

    def describe(self) -> str:
        parts = [f"{r}px x {len(self.folds[r])} folds" for r in self.resolutions]
        return f"{' + '.join(parts)}, TTA {self.tta}, device {self.device}"

    def _model(self, res: int, fold: int):
        key = (res, fold)
        if key not in self._cache:
            self._cache[key] = load_fold(self.specs[res], fold, self.device)
        return self._cache[key]

    def load(self) -> "Predictor":
        """Load every network now instead of on first use (no-op in low-memory mode)."""
        if not self.low_memory:
            for r in self.resolutions:
                for fold in self.folds[r]:
                    self._model(r, fold)
        return self

    def _release(self) -> None:
        self._cache.clear()
        gc.collect()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _prepare(self, points: np.ndarray, path: Optional[str], height: Optional[float]):
        xyz = np.asarray(points[:, :3], dtype=np.float32).astype(np.float64)
        h = float(xyz[:, 2].max() - xyz[:, 2].min()) if height is None else float(height)
        views = {r: render_views(xyz, img_size=r) for r in self.resolutions}
        return {"path": path, "n": len(xyz), "height": h, "views": views,
                "tensors": {r: _to_tensors(v, r) for r, v in views.items()}}

    def _run(self, items: List[dict], keep_views: bool) -> List[Prediction]:
        acc = [{r: [] for r in self.resolutions} for _ in items]
        for r in self.resolutions:
            for fold in self.folds[r]:
                model = self._model(r, fold)
                rng_seed = (self.seed, r, fold)
                for i, it in enumerate(items):
                    rng = np.random.default_rng(rng_seed)
                    acc[i][r].append(tta_probs(model, it["tensors"][r], it["height"], self.tta, rng,
                                               self.device, self.max_batch))
                if self.low_memory:
                    self._release()
        out = []
        for it, a in zip(items, acc):
            per_res = {r: np.mean(v, axis=0) for r, v in a.items()}
            probs = np.mean(list(per_res.values()), axis=0)
            out.append(Prediction(path=it["path"], n_points=it["n"], tree_height=it["height"],
                                  probs=probs, per_resolution=per_res,
                                  views=it["views"] if keep_views else {}))
        return out

    def predict(self, points: np.ndarray, height: Optional[float] = None,
                keep_views: bool = False) -> Prediction:
        """Classify one tree given an (N, 3) XYZ array in metres (Z up)."""
        return self._run([self._prepare(points, None, height)], keep_views)[0]

    def predict_file(self, path: str, height: Optional[float] = None,
                     keep_views: bool = False) -> Prediction:
        return self._run([self._prepare(read_point_cloud(path), path, height)], keep_views)[0]

    def predict_files(self, paths: Iterable[str], chunk_size: Optional[int] = None,
                      keep_views: bool = False) -> Iterator[Prediction]:
        """Yield one Prediction per file. In low-memory mode trees are processed
        in chunks so each network is loaded once per chunk."""
        paths = list(paths)
        chunk_size = chunk_size or (32 if self.low_memory else 1)
        for start in range(0, len(paths), chunk_size):
            items = [self._prepare(read_point_cloud(p), p, None) for p in paths[start:start + chunk_size]]
            yield from self._run(items, keep_views)
