"""Weight download against a fake Google Drive folder (no network)."""
import hashlib
import json
import os
import sys
import types

import pytest

from mvtree import weights

DRIVE = {  # Drive path -> content; README.md and the stale manifest are not in the local manifest
    "convnextv2_256/config.json": b"{}",
    "convnextv2_256/fold0.safetensors": b"fold0",
    "README.md": b"drive readme",
    "manifest.json": b"{}",
}


@pytest.fixture
def fake_gdown(monkeypatch):
    calls = []

    def download_folder(url, output, quiet, skip_download):
        assert skip_download
        return [types.SimpleNamespace(id=path, path=path) for path in DRIVE]

    def download(id, output, quiet):
        calls.append(id)
        with open(output, "wb") as fh:
            fh.write(DRIVE[id])

    monkeypatch.setitem(sys.modules, "gdown", types.SimpleNamespace(download_folder=download_folder,
                                                                    download=download))
    return calls


def write_manifest(root, files):
    entries = {rel: {"bytes": len(DRIVE[rel]), "sha256": hashlib.sha256(DRIVE[rel]).hexdigest()} for rel in files}
    with open(os.path.join(root, "manifest.json"), "w") as fh:
        json.dump({"gdrive_folder": "https://drive.example/folder", "files": entries}, fh)


def test_download_fetches_only_manifest_files(tmp_path, fake_gdown):
    write_manifest(tmp_path, ["convnextv2_256/config.json", "convnextv2_256/fold0.safetensors"])
    weights.download(root=str(tmp_path))
    assert sorted(fake_gdown) == ["convnextv2_256/config.json", "convnextv2_256/fold0.safetensors"]
    assert not (tmp_path / "README.md").exists()
    assert json.loads((tmp_path / "manifest.json").read_text())["gdrive_folder"] == "https://drive.example/folder"
    assert weights.verify(str(tmp_path), quiet=True)


def test_download_skips_verified_files(tmp_path, fake_gdown):
    write_manifest(tmp_path, ["convnextv2_256/config.json", "convnextv2_256/fold0.safetensors"])
    (tmp_path / "convnextv2_256").mkdir()
    (tmp_path / "convnextv2_256" / "config.json").write_bytes(DRIVE["convnextv2_256/config.json"])
    (tmp_path / "convnextv2_256" / "fold0.safetensors").write_bytes(b"corrupt")
    weights.download(root=str(tmp_path))
    assert fake_gdown == ["convnextv2_256/fold0.safetensors"]
    assert weights.verify(str(tmp_path), quiet=True)
