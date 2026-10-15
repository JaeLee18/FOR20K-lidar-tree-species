#!/usr/bin/env python3
"""Download FOR-species20K (Puliti et al.) from Zenodo record 13255198.

Fetches ``dev.zip`` (25 GB), ``test.zip`` (1.8 GB), and ``tree_metadata_dev.csv``
into ``data/raw`` and ``data/metadata``. The zips do not need to be extracted:
``scripts/data/render_dataset.py`` and ``scripts/eval/predict_testset.py`` read scans
directly from them.

    python scripts/data/download_data.py                 # everything
    python scripts/data/download_data.py --only test     # just the hidden-test scans
"""

import argparse
import hashlib
import os

import requests
from tqdm import tqdm

RECORD = "13255198"
API = f"https://zenodo.org/api/records/{RECORD}"
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def md5sum(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: str, size: int) -> None:
    tmp = dest + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        mode = "ab" if done and r.status_code == 206 else "wb"
        with open(tmp, mode) as fh, tqdm(total=size, initial=done if mode == "ab" else 0,
                                          unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for block in r.iter_content(1 << 20):
                fh.write(block)
                bar.update(len(block))
    os.replace(tmp, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["dev", "test", "metadata"], help="download a single file group")
    ap.add_argument("--root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--no-verify", action="store_true", help="skip the md5 check")
    args = ap.parse_args()

    record = requests.get(API, timeout=60).json()
    for f in record["files"]:
        key = f["key"]
        group = "metadata" if key.endswith(".csv") else key.split(".")[0]
        if args.only and group != args.only:
            continue
        dest = os.path.join(args.root, "metadata" if key.endswith(".csv") else "raw", key)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        expected = f.get("checksum", "").replace("md5:", "")
        if os.path.exists(dest) and os.path.getsize(dest) == f["size"]:
            print(f"{key}: already present")
        else:
            fetch(f["links"]["self"], dest, f["size"])
        if expected and not args.no_verify:
            ok = md5sum(dest) == expected
            print(f"{key}: md5 {'OK' if ok else 'MISMATCH'}")
            if not ok:
                raise SystemExit(f"checksum mismatch for {dest}; delete it and re-run")
    print("done. Dataset license: GPL-3.0-or-later (see the Zenodo record).")


if __name__ == "__main__":
    main()
