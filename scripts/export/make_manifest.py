#!/usr/bin/env python3
"""Write weights/manifest.json: the SHA-256 and size of every released file.

Run after exporting weights and before uploading them; commit the manifest.
Set the Google Drive folder link with --gdrive-folder (kept on later runs).

    python scripts/export/make_manifest.py --gdrive-folder https://drive.google.com/drive/folders/<id>
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from mvtree.weights import MANIFEST, read_manifest, sha256, weights_root  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--gdrive-folder", default=None)
    args = ap.parse_args()

    root = weights_root(args.weights)
    old = read_manifest(root)
    files = {}
    for sub in sorted(os.listdir(root)):
        d = os.path.join(root, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            rel = f"{sub}/{f}"
            files[rel] = {"bytes": os.path.getsize(os.path.join(d, f)), "sha256": sha256(os.path.join(d, f))}
            print(f"{rel:40s} {files[rel]['bytes'] / 1e6:9.1f} MB  {files[rel]['sha256'][:16]}")
    manifest = {"gdrive_folder": args.gdrive_folder or old.get("gdrive_folder", ""), "files": files}
    with open(os.path.join(root, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {os.path.join(root, MANIFEST)}")


if __name__ == "__main__":
    main()
