"""SQLite store for the web app's prediction records.

One row per uploaded scan. Predictions are written by the worker thread;
``site``, ``notes`` and ``verified_species`` are edited by people in the UI.
Class probabilities are stored as a JSON list in ``mvtree.constants.SPECIES``
order, so a record stays readable without the model.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from ..constants import GENUS, SPECIES

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT    NOT NULL,
    filename         TEXT    NOT NULL,
    file_size        INTEGER NOT NULL,
    scan_path        TEXT,
    site             TEXT    NOT NULL DEFAULT '',
    notes            TEXT    NOT NULL DEFAULT '',
    verified_species TEXT    NOT NULL DEFAULT '',
    status           TEXT    NOT NULL DEFAULT 'queued',
    error            TEXT,
    started_at       TEXT,
    finished_at      TEXT,
    duration_s       REAL,
    model            TEXT,
    n_points         INTEGER,
    tree_height      REAL,
    species          TEXT,
    confidence       REAL,
    probs            TEXT,
    per_resolution   TEXT
);
CREATE INDEX IF NOT EXISTS idx_records_species ON records(species);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
"""

EDITABLE = {"site": 200, "notes": 5000, "verified_species": 100}
SORTS = {
    "new": "id DESC",
    "old": "id ASC",
    "confidence": "confidence IS NULL, confidence ASC, id DESC",
    "-confidence": "confidence IS NULL, confidence DESC, id DESC",
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RecordStore:
    """Thread-safe wrapper around one SQLite connection."""

    def __init__(self, path: str) -> None:
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.executescript(SCHEMA)

    def _exec(self, sql: str, params: Tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, params)

    def _all(self, sql: str, params: Tuple = ()) -> List[sqlite3.Row]:
        # Fetch under the lock too: the connection is shared, and another thread's
        # execute() would reset a cursor that is still being read.
        with self._lock:
            return self._db.execute(sql, params).fetchall()

    def _one(self, sql: str, params: Tuple = ()) -> Optional[sqlite3.Row]:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # --- writes -------------------------------------------------------------
    def create(self, filename: str, file_size: int, site: str = "") -> int:
        cur = self._exec("INSERT INTO records (created_at, filename, file_size, site) VALUES (?, ?, ?, ?)",
                         (utc_now(), filename, int(file_size), site[:EDITABLE["site"]]))
        return int(cur.lastrowid)

    def set_scan_path(self, rid: int, scan_path: Optional[str]) -> None:
        self._exec("UPDATE records SET scan_path = ? WHERE id = ?", (scan_path, rid))

    def mark_queued(self, rid: int) -> None:
        self._exec("UPDATE records SET status = 'queued', error = NULL, started_at = NULL, finished_at = NULL, "
                   "duration_s = NULL WHERE id = ?", (rid,))

    def mark_running(self, rid: int, model: str) -> None:
        self._exec("UPDATE records SET status = 'running', started_at = ?, model = ? WHERE id = ?",
                   (utc_now(), model, rid))

    def finish(self, rid: int, *, n_points: int, tree_height: float, probs, per_resolution: Dict[int, Any],
               duration_s: float) -> None:
        probs = [round(float(p), 6) for p in probs]
        best = max(range(len(probs)), key=probs.__getitem__)
        per_res = {str(r): [round(float(p), 6) for p in v] for r, v in per_resolution.items()}
        self._exec(
            "UPDATE records SET status = 'done', error = NULL, finished_at = ?, duration_s = ?, n_points = ?, "
            "tree_height = ?, species = ?, confidence = ?, probs = ?, per_resolution = ? WHERE id = ?",
            (utc_now(), round(duration_s, 2), int(n_points), round(float(tree_height), 3), SPECIES[best],
             probs[best], json.dumps(probs), json.dumps(per_res), rid))

    def fail(self, rid: int, error: str) -> None:
        self._exec("UPDATE records SET status = 'error', error = ?, finished_at = ? WHERE id = ?",
                   (error[:2000], utc_now(), rid))

    def update(self, rid: int, fields: Dict[str, Any]) -> None:
        sets, params = [], []
        for key, limit in EDITABLE.items():
            if key in fields:
                value = fields[key]
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
                sets.append(f"{key} = ?")
                params.append(value.strip()[:limit])
        if sets:
            self._exec(f"UPDATE records SET {', '.join(sets)} WHERE id = ?", (*params, rid))

    def delete(self, rid: int) -> None:
        self._exec("DELETE FROM records WHERE id = ?", (rid,))

    # --- reads --------------------------------------------------------------
    def get(self, rid: int, with_probs: bool = True) -> Optional[dict]:
        row = self._one("SELECT * FROM records WHERE id = ?", (rid,))
        return to_dict(row, with_probs) if row else None

    def unfinished(self) -> List[dict]:
        rows = self._all("SELECT * FROM records WHERE status IN ('queued', 'running') ORDER BY id")
        return [to_dict(r, False) for r in rows]

    def list(self, q: str = "", species: str = "", status: str = "", verified: str = "", sort: str = "new",
             limit: int = 50, offset: int = 0, with_probs: bool = False) -> Tuple[List[dict], int]:
        where, params = [], []
        if q:
            where.append("(filename LIKE ? OR site LIKE ? OR notes LIKE ? OR verified_species LIKE ?)")
            params += [f"%{q}%"] * 4
        if species:
            where.append("(species = ? OR verified_species = ?)")
            params += [species, species]
        if status:
            where.append("status = ?")
            params.append(status)
        if verified == "yes":
            where.append("verified_species != ''")
        elif verified == "no":
            where.append("verified_species = ''")
        elif verified == "mismatch":
            where.append("verified_species != '' AND species IS NOT NULL AND verified_species != species")
        clause = f"WHERE {' AND '.join(where)}" if where else ""
        total = self._one(f"SELECT COUNT(*) FROM records {clause}", tuple(params))[0]
        rows = self._all(f"SELECT * FROM records {clause} ORDER BY {SORTS.get(sort, SORTS['new'])} LIMIT ? OFFSET ?",
                         (*params, int(limit), int(offset)))
        return [to_dict(r, with_probs) for r in rows], int(total)

    def stats(self) -> dict:
        row = self._one(
            "SELECT COUNT(*) AS n, SUM(status = 'done') AS done, SUM(status IN ('queued', 'running')) AS pending, "
            "SUM(status = 'error') AS errors, SUM(verified_species != '' AND status = 'done') AS verified, "
            "SUM(verified_species != '' AND status = 'done' AND verified_species = species) AS correct "
            "FROM records")
        by_species = self._all("SELECT species, COUNT(*) AS n FROM records WHERE status = 'done' "
                               "GROUP BY species ORDER BY n DESC, species")
        return {"records": row["n"] or 0, "done": row["done"] or 0, "pending": row["pending"] or 0,
                "errors": row["errors"] or 0, "verified": row["verified"] or 0, "correct": row["correct"] or 0,
                "by_species": [{"species": r["species"], "n": r["n"]} for r in by_species]}


def to_dict(row: sqlite3.Row, with_probs: bool = True, k: int = 5) -> dict:
    rec = {key: row[key] for key in row.keys() if key not in ("probs", "per_resolution", "scan_path")}
    rec["has_scan"] = bool(row["scan_path"])
    rec["genus"] = GENUS.get(row["species"] or "", None)
    verified = row["verified_species"]
    rec["match"] = None if not verified or not row["species"] else verified == row["species"]
    if row["probs"]:
        probs = json.loads(row["probs"])
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        rec["top_k"] = [{"species": SPECIES[i], "prob": probs[i]} for i in order[:k]]
        if with_probs:
            rec["probs"] = dict(zip(SPECIES, probs))
            per_res = json.loads(row["per_resolution"] or "{}")
            rec["per_resolution"] = {
                r: {"species": SPECIES[max(range(len(v)), key=v.__getitem__)], "prob": max(v)}
                for r, v in per_res.items()
            }
    return rec
