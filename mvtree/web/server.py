"""Self-hosted web app: upload scans in a browser and keep a searchable record of every prediction.

    mvtree serve                                   # http://127.0.0.1:8000, records in ./mvtree_data
    mvtree serve --host 0.0.0.0 --token SECRET     # share on a network; the UI asks for the token once

Standard library only (``http.server`` + ``sqlite3``). One worker thread owns the
model and processes uploads in arrival order; request threads only read and write
the database and the per-record files::

    <data-dir>/records.sqlite3
    <data-dir>/records/<id>/scan.<ext>     original upload (unless --no-keep-scans)
    <data-dir>/records/<id>/montage.png    the 7 model-input views
    <data-dir>/records/<id>/thumb.png      one side view, for the table
    <data-dir>/records/<id>/points.f32     <= 20k points (x, y, z float32) for the 3D viewer
"""

from __future__ import annotations

import csv
import io
import json
import os
import queue
import re
import secrets
import shutil
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse

import numpy as np

from .. import __version__
from ..constants import SPECIES
from ..io import SUPPORTED_EXTENSIONS
from .store import RecordStore

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}
ASSETS = {"montage.png": "image/png", "thumb.png": "image/png", "points.f32": "application/octet-stream"}
RECORD_ROUTE = re.compile(r"^/api/records/(\d+)(?:/([a-z0-9_.]+))?$")
VIEWER_POINTS = 20000


def safe_filename(name: str) -> str:
    """Basename only, conservative characters, bounded length."""
    name = os.path.basename(name.replace("\\", "/")).strip() or "scan"
    name = re.sub(r"[^A-Za-z0-9._ ()+-]", "_", name)
    return name[-120:]


class App:
    """Shared state: storage, configuration, and the prediction worker."""

    def __init__(self, data_dir: str, make_predictor: Callable[[], object], keep_scans: bool = True,
                 token: Optional[str] = None, max_upload_mb: float = 500) -> None:
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(os.path.join(self.data_dir, "records"), exist_ok=True)
        self.store = RecordStore(os.path.join(self.data_dir, "records.sqlite3"))
        self.keep_scans = keep_scans
        self.token = token or None
        self.max_upload = int(max_upload_mb * 1024 * 1024)
        self.worker = Worker(self, make_predictor)

    def record_dir(self, rid: int) -> str:
        return os.path.join(self.data_dir, "records", str(rid))

    def scan_path(self, rid: int) -> Optional[str]:
        rec_dir = self.record_dir(rid)
        if not os.path.isdir(rec_dir):
            return None
        for name in os.listdir(rec_dir):
            if name.startswith("scan."):
                return os.path.join(rec_dir, name)
        return None

    def start(self) -> None:
        """Start the worker and re-queue anything a previous run left unfinished."""
        for rec in self.store.unfinished():
            if self.scan_path(rec["id"]):
                self.store.mark_queued(rec["id"])
                self.worker.jobs.put(rec["id"])
            else:
                self.store.fail(rec["id"], "interrupted by a server restart and the scan was not kept")
        self.worker.start()

    def save_assets(self, rid: int, xyz: np.ndarray, pred) -> None:
        from PIL import Image

        from ..viz import montage, on_white

        rec_dir = self.record_dir(rid)
        if not os.path.isdir(rec_dir):  # deleted while running
            return
        views = pred.views[min(pred.views)]
        montage(views, tile=160).save(os.path.join(rec_dir, "montage.png"))
        Image.fromarray(on_white(views["side_000"])).resize((112, 112), Image.NEAREST).save(
            os.path.join(rec_dir, "thumb.png"))
        pts = np.asarray(xyz, dtype=np.float64)
        if len(pts) > VIEWER_POINTS:
            pts = pts[np.random.default_rng(0).choice(len(pts), VIEWER_POINTS, replace=False)]
        pts = pts - [pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].min()]
        pts.astype("<f4").tofile(os.path.join(rec_dir, "points.f32"))


def _release_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


class Worker(threading.Thread):
    """Loads the model once, then predicts queued records one at a time."""

    def __init__(self, app: App, make_predictor: Callable[[], object]) -> None:
        super().__init__(name="mvtree-worker", daemon=True)
        self.app = app
        self.make_predictor = make_predictor
        self.jobs: "queue.Queue[int]" = queue.Queue()
        self.state, self.model, self.error = "loading", "", None
        self.current: Optional[int] = None
        self.predictor = None

    def run(self) -> None:
        try:
            self.predictor = self.make_predictor()
            if hasattr(self.predictor, "load"):
                # Load every network and run one throwaway tree, so "ready" means the weights are in
                # memory and the first upload does not pay for CUDA warm-up (~20 s on an RTX 4090).
                self.predictor.load()
                rng = np.random.default_rng(0)
                self.predictor.predict(np.c_[rng.normal(0, 1.5, (2000, 2)), rng.uniform(0, 12, 2000)])
                _release_cuda_cache()
            self.model = self.predictor.describe()
            self.state = "ready"
            print(f"[mvtree] model ready: {self.model}", file=sys.stderr)
        except Exception as exc:  # keep serving records; new uploads fail with this message
            self.state, self.error = "error", f"{type(exc).__name__}: {exc}"
            print(f"[mvtree] model failed to load: {self.error}", file=sys.stderr)
        while True:
            rid = self.jobs.get()
            self.current = rid
            try:
                self._process(rid)
            finally:
                self.current = None
                _release_cuda_cache()

    def _predict(self, xyz: np.ndarray):
        """Predict, halving the forward-pass batch on CUDA out-of-memory (results do not depend on it)."""
        import torch

        while True:
            try:
                return self.predictor.predict(xyz, keep_views=True)
            except torch.cuda.OutOfMemoryError:
                batch = getattr(self.predictor, "max_batch", 1)
                if batch <= 1:
                    raise
                torch.cuda.empty_cache()
                self.predictor.max_batch = batch // 2
                print(f"[mvtree] GPU out of memory; retrying with --max-batch {batch // 2}", file=sys.stderr)

    def _process(self, rid: int) -> None:
        from ..io import read_point_cloud

        store = self.app.store
        rec = store.get(rid, with_probs=False)
        scan = self.app.scan_path(rid)
        if rec is None or rec["status"] != "queued":
            return
        if self.state != "ready":
            store.fail(rid, f"model not available: {self.error}")
            return
        if scan is None:
            store.fail(rid, "scan file is missing")
            return
        store.mark_running(rid, self.model)
        t0 = time.time()
        try:
            xyz = read_point_cloud(scan)
            pred = self._predict(xyz)
            self.app.save_assets(rid, xyz, pred)
            store.finish(rid, n_points=pred.n_points, tree_height=pred.tree_height, probs=pred.probs,
                         per_resolution=pred.per_resolution, duration_s=time.time() - t0)
            print(f"[mvtree] record {rid} ({rec['filename']}): {pred.species} {pred.confidence:.1%} "
                  f"in {time.time() - t0:.1f}s", file=sys.stderr)
        except Exception as exc:
            store.fail(rid, f"{type(exc).__name__}: {exc}")
            print(f"[mvtree] record {rid} ({rec['filename']}) failed: {exc}", file=sys.stderr)
        finally:
            if not self.app.keep_scans and os.path.exists(scan):
                os.remove(scan)
                store.set_scan_path(rid, None)


class Handler(BaseHTTPRequestHandler):
    server_version = f"mvtree/{__version__}"
    app: App  # set by make_server

    def log_message(self, fmt, *args) -> None:  # polling would flood the terminal
        pass

    # --- plumbing -----------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, headers: Optional[dict] = None,
              cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(status, json.dumps(obj).encode(), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _body_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 1_000_000:
            raise ValueError("request body too large")
        data = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data

    def _authorized(self, qs: dict) -> bool:
        if not self.app.token:
            return True
        given = self.headers.get("X-Mvtree-Token") or qs.get("token", "")
        return secrets.compare_digest(given.encode(), self.app.token.encode())

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_HEAD(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def do_PATCH(self) -> None:
        self._dispatch("PATCH")

    def do_DELETE(self) -> None:
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        url = urlparse(self.path)
        qs = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            if method == "GET" and url.path in STATIC_FILES:
                name, ctype = STATIC_FILES[url.path]
                with open(os.path.join(STATIC_DIR, name), "rb") as fh:
                    return self._send(200, fh.read(), ctype)
            if not url.path.startswith("/api/"):
                return self._error(404, "not found")
            if not self._authorized(qs):
                self.close_connection = True
                return self._error(401, "access token required")
            if url.path == "/api/status" and method == "GET":
                return self._status()
            if url.path == "/api/records" and method == "GET":
                return self._list(qs)
            if url.path == "/api/records" and method == "POST":
                return self._upload(qs)
            if url.path == "/api/export.csv" and method == "GET":
                return self._export()
            m = RECORD_ROUTE.match(url.path)
            if m:
                return self._record(method, int(m.group(1)), m.group(2))
            return self._error(404, "not found")
        except (BrokenPipeError, ConnectionResetError):
            pass
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to the client and the log
            traceback.print_exc()
            self._error(500, f"{type(exc).__name__}: {exc}")

    # --- endpoints ----------------------------------------------------------
    def _status(self) -> None:
        w = self.app.worker
        self._json({"version": __version__, "state": w.state, "model": w.model, "error": w.error,
                    "queue": w.jobs.qsize(), "running": w.current, "species": SPECIES,
                    "keep_scans": self.app.keep_scans, "max_upload_mb": round(self.app.max_upload / 2**20),
                    "extensions": list(SUPPORTED_EXTENSIONS), "stats": self.app.store.stats()})

    def _list(self, qs: dict) -> None:
        limit = min(max(int(qs.get("limit", 50)), 1), 500)
        records, total = self.app.store.list(q=qs.get("q", "").strip(), species=qs.get("species", ""),
                                             status=qs.get("status", ""), verified=qs.get("verified", ""),
                                             sort=qs.get("sort", "new"), limit=limit,
                                             offset=max(int(qs.get("offset", 0)), 0))
        self._json({"records": records, "total": total})

    def _upload(self, qs: dict) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error(400, "empty upload")
        if length > self.app.max_upload:
            self.close_connection = True
            return self._error(413, f"file is larger than the {self.app.max_upload // 2**20} MB limit")
        filename = safe_filename(qs.get("filename", "scan"))
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            self.close_connection = True
            return self._error(415, f"unsupported file type '{ext or filename}'; use one of "
                                    f"{', '.join(SUPPORTED_EXTENSIONS)}")
        store = self.app.store
        rid = store.create(filename, length, qs.get("site", "").strip())
        rec_dir = self.app.record_dir(rid)
        os.makedirs(rec_dir, exist_ok=True)
        path = os.path.join(rec_dir, f"scan{ext}")
        remaining = length
        with open(path, "wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)
        if remaining:
            store.delete(rid)
            shutil.rmtree(rec_dir, ignore_errors=True)
            return self._error(400, "upload interrupted")
        store.set_scan_path(rid, os.path.relpath(path, self.app.data_dir))
        self.app.worker.jobs.put(rid)
        self._json(store.get(rid), 201)

    def _record(self, method: str, rid: int, sub: Optional[str]) -> None:
        store = self.app.store
        rec = store.get(rid)
        if rec is None:
            return self._error(404, f"record {rid} not found")
        if sub is None and method == "GET":
            return self._json(rec)
        if sub is None and method == "PATCH":
            store.update(rid, self._body_json())
            return self._json(store.get(rid))
        if sub is None and method == "DELETE":
            if rec["status"] == "running":
                return self._error(409, "record is being predicted; delete it when it finishes")
            store.delete(rid)
            shutil.rmtree(self.app.record_dir(rid), ignore_errors=True)
            return self._json({"deleted": rid})
        if sub == "rerun" and method == "POST":
            if rec["status"] in ("queued", "running"):
                return self._error(409, "record is already queued")
            if not self.app.scan_path(rid):
                return self._error(409, "the original scan was not kept, so it cannot be re-run")
            store.mark_queued(rid)
            self.app.worker.jobs.put(rid)
            return self._json(store.get(rid))
        if sub == "scan" and method == "GET":
            path = self.app.scan_path(rid)
            if not path:
                return self._error(404, "the original scan was not kept")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(os.path.getsize(path)))
            self.send_header("Content-Disposition", f'attachment; filename="{rec["filename"]}"')
            self.end_headers()
            if self.command != "HEAD":
                with open(path, "rb") as fh:
                    shutil.copyfileobj(fh, self.wfile, 1 << 20)
            return
        if sub in ASSETS and method == "GET":
            path = os.path.join(self.app.record_dir(rid), sub)
            if not os.path.isfile(path):
                return self._error(404, f"{sub} not available yet")
            with open(path, "rb") as fh:  # the UI versions these URLs by finish time, so caching is safe
                return self._send(200, fh.read(), ASSETS[sub], cache="private, max-age=86400")
        return self._error(404, "not found")

    def _export(self) -> None:
        records, _ = self.app.store.list(limit=10**9, sort="old")
        buf = io.StringIO()
        w = csv.writer(buf)
        header = ["id", "created_at_utc", "filename", "site", "status", "n_points", "tree_height_m", "species",
                  "genus", "confidence"]
        for k in range(2, 6):
            header += [f"top{k}_species", f"top{k}_prob"]
        header += ["verified_species", "match", "notes", "model", "duration_s", "error"]
        w.writerow(header)
        for r in records:
            row = [r["id"], r["created_at"], r["filename"], r["site"], r["status"], r["n_points"], r["tree_height"],
                   r["species"], r["genus"], r["confidence"]]
            top = r.get("top_k", [])[1:5]
            for k in range(4):
                row += [top[k]["species"], f"{top[k]['prob']:.4f}"] if k < len(top) else ["", ""]
            row += [r["verified_species"], "" if r["match"] is None else int(r["match"]), r["notes"], r["model"],
                    r["duration_s"], r["error"]]
            w.writerow(row)
        name = f"mvtree_records_{time.strftime('%Y%m%d')}.csv"
        self._send(200, buf.getvalue().encode(), "text/csv; charset=utf-8",
                   {"Content-Disposition": f'attachment; filename="{name}"'})


def make_server(host: str, port: int, app: App) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(host: str, port: int, data_dir: str, make_predictor: Callable[[], object], keep_scans: bool = True,
          token: Optional[str] = None, max_upload_mb: float = 500) -> int:
    app = App(data_dir, make_predictor, keep_scans=keep_scans, token=token, max_upload_mb=max_upload_mb)
    httpd = make_server(host, port, app)
    app.start()
    shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "::") else host
    print(f"[mvtree] web app on http://{shown}:{httpd.server_port}  (records in {app.data_dir})", file=sys.stderr)
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("[mvtree] warning: listening on the network without --token; anyone who can reach this port "
              "can upload and delete records", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[mvtree] stopped", file=sys.stderr)
    finally:
        httpd.server_close()
    return 0
