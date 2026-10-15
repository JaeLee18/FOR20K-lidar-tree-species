"""Point-cloud readers.

Every reader returns an (N, 3) float32 XYZ array (metres, Z up). Coordinates
are rounded to float32 on purpose: the training renders were produced from
float32 clouds, and the renderer's min-max normalisation is sensitive to that
rounding at the pixel boundary.

Supported: .las/.laz (laspy + lazrs), .ply (ASCII or binary), .pcd (ASCII or
binary, uncompressed), .npy/.npz, and whitespace/comma text (.xyz/.txt/.csv/.pts,
first three numeric columns).
"""

from __future__ import annotations

import os
from typing import Iterable, List

import numpy as np

SUPPORTED_EXTENSIONS = (".las", ".laz", ".ply", ".pcd", ".npy", ".npz", ".xyz", ".txt", ".csv", ".pts")


def _as_xyz(arr: np.ndarray, source: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"{source}: expected an (N, >=3) array, got shape {arr.shape}")
    xyz = arr[:, :3].astype(np.float32)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if xyz.shape[0] < 3:
        raise ValueError(f"{source}: fewer than 3 finite points")
    return xyz


def read_las(path: str) -> np.ndarray:
    import laspy

    las = laspy.read(path)
    return _as_xyz(np.stack([las.x, las.y, las.z], axis=-1), path)


def read_ply(path: str) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Reading .ply needs `pip install plyfile`") from exc
    vertex = PlyData.read(path)["vertex"]
    return _as_xyz(np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1), path)


def read_pcd(path: str) -> np.ndarray:
    """Minimal PCD reader (ASCII and binary, uncompressed, fields x y z first-class)."""
    with open(path, "rb") as fh:
        header = {}
        while True:
            line = fh.readline().decode("ascii", errors="ignore").strip()
            if not line or line.startswith("#"):
                continue
            key, *vals = line.split()
            header[key.upper()] = vals
            if key.upper() == "DATA":
                break
        fields = header["FIELDS"]
        sizes = [int(s) for s in header["SIZE"]]
        types = header["TYPE"]
        counts = [int(c) for c in header.get("COUNT", ["1"] * len(fields))]
        n = int(header["POINTS"][0])
        mode = header["DATA"][0].lower()
        if mode == "ascii":
            data = np.loadtxt(fh, dtype=np.float64, ndmin=2)
            cols = []
            offset = 0
            for f, c in zip(fields, counts):
                cols.append((f, offset))
                offset += c
            idx = {f: o for f, o in cols}
            return _as_xyz(data[:, [idx["x"], idx["y"], idx["z"]]], path)
        if mode != "binary":
            raise ValueError(f"{path}: PCD DATA '{mode}' not supported (use ascii or binary)")
        kinds = {"F": "f", "I": "i", "U": "u"}
        dtype = np.dtype([(f, f"<{kinds[t]}{s}", (c,)) if c > 1 else (f, f"<{kinds[t]}{s}")
                          for f, s, t, c in zip(fields, sizes, types, counts)])
        data = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype, count=n)
        return _as_xyz(np.stack([data["x"], data["y"], data["z"]], axis=-1), path)


def read_numpy(path: str) -> np.ndarray:
    if path.endswith(".npz"):
        with np.load(path) as z:
            key = next((k for k in ("points", "xyz", "pos", "coord") if k in z.files), z.files[0])
            return _as_xyz(z[key], path)
    return _as_xyz(np.load(path), path)


def read_text(path: str) -> np.ndarray:
    delimiter = "," if path.lower().endswith(".csv") else None
    rows = []
    with open(path, "r") as fh:
        for line in fh:
            parts = line.replace(",", " ").split() if delimiter else line.split()
            if len(parts) < 3:
                continue
            try:
                rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue  # header or comment line
    return _as_xyz(np.asarray(rows, dtype=np.float64), path)


def read_point_cloud(path: str) -> np.ndarray:
    """Read a single-tree point cloud from any supported format."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".las", ".laz"):
        return read_las(path)
    if ext == ".ply":
        return read_ply(path)
    if ext == ".pcd":
        return read_pcd(path)
    if ext in (".npy", ".npz"):
        return read_numpy(path)
    if ext in (".xyz", ".txt", ".csv", ".pts"):
        return read_text(path)
    raise ValueError(f"unsupported point-cloud format '{ext}' ({path}); "
                     f"supported: {', '.join(SUPPORTED_EXTENSIONS)}")


def find_point_clouds(inputs: Iterable[str]) -> List[str]:
    """Expand files and directories into a sorted list of point-cloud paths."""
    paths: List[str] = []
    for item in inputs:
        if os.path.isdir(item):
            for root, _, files in os.walk(item):
                paths += [os.path.join(root, f) for f in files
                          if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]
        elif os.path.isfile(item):
            paths.append(item)
        else:
            raise FileNotFoundError(item)
    return sorted(paths)
