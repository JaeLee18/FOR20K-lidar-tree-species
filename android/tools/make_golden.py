#!/usr/bin/env python3
"""Write golden test data for the Kotlin core from the Python reference implementation.

    python android/tools/make_golden.py [--model android/app/src/main/assets/model.onnx]

For each example scan: the reader output (point count + SHA-256 of the float32
XYZ bytes), the tree height, the seven 256 px renders as PNG, and, with
--model, the int8 model's class probabilities (ONNX Runtime, CPU). Also writes a
small synthetic cloud in every supported format with the reader's expected
output, so each Kotlin reader is checked against `mvtree.io`.
"""

import argparse
import glob
import hashlib
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
from mvtree.constants import SPECIES, render_stats  # noqa: E402
from mvtree.io import read_point_cloud  # noqa: E402
from mvtree.render import DETAIL_VIEW, SIDE_VIEWS, TOPBOT_VIEWS, render_views, save_views  # noqa: E402

RES = os.path.join(ROOT, "android", "core", "src", "test", "resources")


def digest(xyz: np.ndarray) -> dict:
    xyz = np.ascontiguousarray(xyz, dtype="<f4")
    return {"n": int(len(xyz)), "sha256": hashlib.sha256(xyz.tobytes()).hexdigest()}


def model_inputs(views, height, size):
    st = render_stats(size)
    mean = np.asarray(st["mean"], np.float32).reshape(3, 1, 1)
    std = np.asarray(st["std"], np.float32).reshape(3, 1, 1)

    def norm(name):
        return (views[name].transpose(2, 0, 1).astype(np.float32) / np.float32(255) - mean) / std

    return {"side_views": np.stack([norm(v) for v in SIDE_VIEWS])[None],
            "topbot_views": np.stack([norm(v) for v in TOPBOT_VIEWS])[None],
            "detail_view": norm(DETAIL_VIEW)[None],
            "tree_height": np.array([[height]], dtype=np.float32)}


def write_formats(out: str) -> dict:
    """One synthetic tree in every format, plus what mvtree.io reads from each file."""
    import laspy

    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(7)
    trunk = np.c_[rng.normal(0, 0.15, (400, 2)), rng.uniform(0, 5, 400)]
    crown = np.c_[rng.normal(0, 2.0, (1600, 2)), rng.uniform(4, 14, 1600)]
    pts = np.vstack([trunk, crown]) + [523401.123456, 4182913.654321, 112.5]  # UTM-like offsets
    files = {}

    np.savetxt(os.path.join(out, "tree.xyz"), pts, fmt="%.6f")
    with open(os.path.join(out, "tree.csv"), "w") as fh:
        fh.write("x,y,z,intensity\n")
        for p in pts:
            fh.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},7\n")
    np.save(os.path.join(out, "tree.npy"), pts)
    np.savez(os.path.join(out, "tree.npz"), labels=np.zeros(3), points=pts.astype(np.float32))

    with open(os.path.join(out, "tree_ascii.ply"), "w") as fh:
        fh.write(f"ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty double x\nproperty double y\n"
                 "property double z\nproperty uchar red\nend_header\n")
        for p in pts:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 200\n")
    with open(os.path.join(out, "tree_binary.ply"), "wb") as fh:
        fh.write(f"ply\nformat binary_little_endian 1.0\ncomment synthetic\nelement vertex {len(pts)}\n"
                 "property float x\nproperty float y\nproperty float z\nproperty ushort intensity\n"
                 "element face 0\nproperty list uchar int vertex_indices\nend_header\n".encode())
        for p in pts:
            fh.write(struct.pack("<fffH", *p.astype(np.float32), 9))

    hdr = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 8 8 8 4\nTYPE F F F U\nCOUNT 1 1 1 1\n"
           f"WIDTH {len(pts)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(pts)}\n")
    with open(os.path.join(out, "tree_ascii.pcd"), "w") as fh:
        fh.write(hdr + "DATA ascii\n")
        for p in pts:
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 3\n")
    with open(os.path.join(out, "tree_binary.pcd"), "wb") as fh:
        fh.write((hdr + "DATA binary\n").encode())
        for p in pts:
            fh.write(struct.pack("<dddI", *p, 3))

    for name, version, fmt in [("tree_v12.las", "1.2", 3), ("tree_v14.las", "1.4", 6)]:
        las = laspy.create(point_format=fmt, file_version=version)
        las.header.offsets = pts.min(axis=0)
        las.header.scales = [0.001, 0.001, 0.001]
        las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
        las.write(os.path.join(out, name))
    las = laspy.read(os.path.join(out, "tree_v12.las"))
    las.write(os.path.join(out, "tree_v12.laz"))

    for path in sorted(glob.glob(os.path.join(out, "tree*"))):
        files[os.path.basename(path)] = digest(read_point_cloud(path))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="int8 model.onnx; adds expected probabilities")
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    session = None
    if args.model:
        import onnxruntime as ort
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    golden = {"size": args.size, "species": SPECIES, "examples": {}}
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "*.laz"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        xyz = read_point_cloud(path)  # float32, exactly what the Predictor uses
        pts = xyz.astype(np.float64)
        height = float(pts[:, 2].max() - pts[:, 2].min())
        views = render_views(pts, img_size=args.size)
        save_views(views, os.path.join(RES, "golden", stem))
        entry = {**digest(xyz), "height": height}
        if session is not None:
            probs = session.run(None, model_inputs(views, height, args.size))[0][0]
            entry["probs"] = [float(p) for p in probs]
            entry["species"] = SPECIES[int(probs.argmax())]
        golden["examples"][os.path.basename(path)] = entry
        print(stem, entry.get("species", ""), entry["n"])
    golden["formats"] = write_formats(os.path.join(RES, "formats"))
    with open(os.path.join(RES, "golden", "golden.json"), "w") as fh:
        json.dump(golden, fh, indent=1)
    print(f"wrote {RES}/golden and {RES}/formats")


if __name__ == "__main__":
    main()
