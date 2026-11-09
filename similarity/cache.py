"""The similarity cache: model-built matrices and endpoint vectors, in one SQLite file.

One file (one inode, safe for parallel jobs in WAL mode), two tables:

  * `matrices`: a scorer's whole matrix for one lecture, under `matrix_key`: the sha256
    of the scorer, every setting that changes its numbers - its own and those of every
    scorer it is built from - the unit texts, the slide texts and the digest of every
    slide image it reads.
  * `vectors`: one embedding per text, under the sha256 of (model, the exact string
    sent). Truncation and any instruction prefix are part of that string, so they are in
    the key without being listed.

Every setting that changes the numbers is in the key: the model of the vectors and of
the image matrix, and for the cross-encoder matrix also the settings of the hybrid
matrix it reranks (the embedding model, the BM25 settings, the blend weight). A key
without them would return another model's or another first stage's numbers.

Values are stored as raw float64 (matrices) or float32 (vectors) bytes, so a matrix read
back is bit-identical to the one that was written.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def digest(obj: Any) -> str:
    """sha256 of the canonical JSON of `obj` (sorted keys, no whitespace variation)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def vector_key(model: str, text: str) -> str:
    return digest(["embedding", model, text])


class SimilarityCache:
    def __init__(self, path: Path | str):
        p = Path(path)
        self.path = p if p.is_absolute() else REPO / p
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self, create: bool) -> Optional[sqlite3.Connection]:
        if self._conn is None:
            if not create and not self.path.exists():
                return None
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=60, isolation_level=None)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("CREATE TABLE IF NOT EXISTS matrices (key TEXT PRIMARY KEY, "
                               "scorer TEXT NOT NULL, n INTEGER NOT NULL, m INTEGER NOT NULL, "
                               "data BLOB NOT NULL)")
            self._conn.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, "
                               "dim INTEGER NOT NULL, data BLOB NOT NULL)")
        return self._conn

    # ------------------------------------------------------------------ matrices

    def get_matrix(self, key: str) -> Optional[np.ndarray]:
        conn = self._connect(create=False)
        row = conn.execute("SELECT n, m, data FROM matrices WHERE key=?", (key,)).fetchone() \
            if conn is not None else None
        if row is None:
            return None
        n, m, data = row
        return np.frombuffer(data, dtype=np.float64).reshape(n, m).copy()

    def put_matrix(self, key: str, scorer: str, M: np.ndarray) -> None:
        M = np.ascontiguousarray(M, dtype=np.float64)
        self._connect(create=True).execute(
            "INSERT OR REPLACE INTO matrices (key, scorer, n, m, data) VALUES (?, ?, ?, ?, ?)",
            (key, scorer, int(M.shape[0]), int(M.shape[1]), M.tobytes()))

    # ------------------------------------------------------------------ vectors

    def get_vectors(self, keys: Iterable[str]) -> dict[str, np.ndarray]:
        conn = self._connect(create=False)
        out: dict[str, np.ndarray] = {}
        if conn is None:
            return out
        keys = list(dict.fromkeys(keys))
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = "SELECT key, data FROM vectors WHERE key IN (%s)" % ",".join("?" * len(part))
            for k, data in conn.execute(q, part):
                out[k] = np.frombuffer(data, dtype=np.float32).copy()
        return out

    def put_vectors(self, items: dict[str, np.ndarray]) -> None:
        conn = self._connect(create=True)
        conn.execute("BEGIN")
        for k, v in items.items():
            v = np.ascontiguousarray(v, dtype=np.float32).ravel()
            conn.execute("INSERT OR REPLACE INTO vectors (key, dim, data) VALUES (?, ?, ?)",
                         (k, int(v.size), v.tobytes()))
        conn.execute("COMMIT")

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


__all__ = ["SimilarityCache", "digest", "vector_key"]
