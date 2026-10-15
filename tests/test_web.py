"""Web app (``mvtree serve``): real HTTP server, stand-in predictor, no weights needed."""
import json
import os
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from mvtree.constants import SPECIES
from mvtree.predict import Prediction
from mvtree.render import render_views
from mvtree.web.server import App, make_server, safe_filename


class FakePredictor:
    """Renders real views (so previews are exercised) but returns fixed probabilities."""

    def describe(self):
        return "fake 64px, TTA 1, device cpu"

    def predict(self, xyz, keep_views=False):
        probs = np.full(len(SPECIES), 0.2 / (len(SPECIES) - 1))
        probs[SPECIES.index("Picea_abies")] = 0.8
        xyz = np.asarray(xyz, dtype=np.float64)
        return Prediction(path=None, n_points=len(xyz), tree_height=float(np.ptp(xyz[:, 2])), probs=probs,
                          per_resolution={64: probs}, views={64: render_views(xyz, img_size=64)} if keep_views else {})


def start(tmp_path, **kw):
    app = App(str(tmp_path / "data"), FakePredictor, **kw)
    httpd = make_server("127.0.0.1", 0, app)
    app.start()
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return app, httpd, f"http://127.0.0.1:{httpd.server_port}"


def call(url, method="GET", body=None, headers=None, raw=False):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            payload = res.read()
            return res.status, (payload if raw else json.loads(payload or b"null")), res.headers
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read() or b"null"), err.headers


def wait_done(base, rid, timeout=60, headers=None):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, rec, _ = call(f"{base}/api/records/{rid}", headers=headers)
        if rec["status"] not in ("queued", "running"):
            return rec
        time.sleep(0.1)
    raise TimeoutError(rid)


def xyz_bytes(cloud):
    return "\n".join(" ".join(f"{v:.4f}" for v in row) for row in cloud).encode()


@pytest.fixture
def server(tmp_path):
    app, httpd, base = start(tmp_path)
    yield app, base
    httpd.shutdown()


def test_upload_predict_edit_export_delete(server, cloud):
    app, base = server
    status, html, _ = call(f"{base}/", raw=True)
    assert status == 200 and b"mvtree" in html

    status, rec, _ = call(f"{base}/api/records?filename=../../tree%2001.xyz&site=Plot%207", "POST", xyz_bytes(cloud))
    assert status == 201 and rec["status"] == "queued" and rec["filename"] == "tree 01.xyz" and rec["site"] == "Plot 7"
    rec = wait_done(base, rec["id"])
    assert rec["status"] == "done", rec["error"]
    assert rec["species"] == "Picea_abies" and rec["genus"] == "Picea" and rec["n_points"] == len(cloud)
    assert rec["top_k"][0] == {"species": "Picea_abies", "prob": 0.8} and abs(sum(rec["probs"].values()) - 1) < 1e-4
    assert rec["has_scan"] and rec["per_resolution"]["64"]["species"] == "Picea_abies"

    for asset, ctype in [("thumb.png", "image/png"), ("montage.png", "image/png")]:
        status, body, headers = call(f"{base}/api/records/{rec['id']}/{asset}", raw=True)
        assert status == 200 and headers["Content-Type"] == ctype and body[:4] == b"\x89PNG"
    status, pts, _ = call(f"{base}/api/records/{rec['id']}/points.f32", raw=True)
    assert status == 200 and len(pts) == 12 * len(cloud)
    status, scan, _ = call(f"{base}/api/records/{rec['id']}/scan", raw=True)
    assert scan == xyz_bytes(cloud)

    edit = {"verified_species": "Picea_abies", "notes": "DBH 41 cm"}
    status, rec2, _ = call(f"{base}/api/records/{rec['id']}", "PATCH", edit, {"Content-Type": "application/json"})
    assert status == 200 and rec2["match"] is True and rec2["notes"] == "DBH 41 cm"
    _, st, _ = call(f"{base}/api/status")
    assert st["state"] == "ready" and st["stats"]["verified"] == 1 and st["stats"]["correct"] == 1

    _, listing, _ = call(f"{base}/api/records?q=DBH")
    assert listing["total"] == 1
    _, listing, _ = call(f"{base}/api/records?verified=mismatch")
    assert listing["total"] == 0

    status, csv_bytes, headers = call(f"{base}/api/export.csv", raw=True)
    lines = csv_bytes.decode().strip().splitlines()
    assert status == 200 and "attachment" in headers["Content-Disposition"]
    assert lines[0].startswith("id,created_at_utc,filename") and "Picea_abies" in lines[1] and "DBH 41 cm" in lines[1]

    status, rerun, _ = call(f"{base}/api/records/{rec['id']}/rerun", "POST", b"")
    assert status == 200 and rerun["status"] == "queued"
    assert wait_done(base, rec["id"])["status"] == "done"

    status, _, _ = call(f"{base}/api/records/{rec['id']}", "DELETE")
    assert status == 200
    assert call(f"{base}/api/records/{rec['id']}")[0] == 404
    assert not os.path.exists(app.record_dir(rec["id"]))


def test_rejects_bad_uploads(server):
    _, base = server
    assert call(f"{base}/api/records?filename=photo.jpg", "POST", b"abc")[0] == 415
    status, rec, _ = call(f"{base}/api/records?filename=broken.xyz", "POST", b"not numbers at all")
    assert status == 201
    rec = wait_done(base, rec["id"])
    assert rec["status"] == "error" and rec["error"]
    assert call(f"{base}/api/records/999")[0] == 404


def test_token_and_no_keep_scans(tmp_path, cloud):
    app, httpd, base = start(tmp_path, token="s3cret", keep_scans=False)
    try:
        assert call(f"{base}/", raw=True)[0] == 200  # the page itself is public; the API is not
        assert call(f"{base}/api/status")[0] == 401
        assert call(f"{base}/api/status", headers={"X-Mvtree-Token": "wrong"})[0] == 401
        assert call(f"{base}/api/status?token=s3cret")[0] == 200
        auth = {"X-Mvtree-Token": "s3cret"}
        status, rec, _ = call(f"{base}/api/records?filename=t.xyz", "POST", xyz_bytes(cloud), auth)
        assert status == 201
        rec = wait_done(base, rec["id"], headers=auth)
        assert rec["status"] == "done" and not rec["has_scan"]
        assert not os.path.exists(os.path.join(app.record_dir(rec["id"]), "scan.xyz"))
        assert call(f"{base}/api/records/{rec['id']}/rerun", "POST", b"", auth)[0] == 409
    finally:
        httpd.shutdown()


def test_restart_requeues_unfinished(tmp_path, cloud):
    app, httpd, base = start(tmp_path)
    httpd.shutdown()
    rid = app.store.create("left_over.xyz", 10)  # as if the server died before predicting it
    os.makedirs(app.record_dir(rid), exist_ok=True)
    with open(os.path.join(app.record_dir(rid), "scan.xyz"), "wb") as fh:
        fh.write(xyz_bytes(cloud))
    app2, httpd2, base2 = start(tmp_path)
    try:
        assert wait_done(base2, rid)["status"] == "done"
    finally:
        httpd2.shutdown()


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("C:\\scans\\tree<1>.laz") == "tree_1_.laz"
    assert safe_filename("") == "scan"
