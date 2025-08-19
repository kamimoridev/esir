from __future__ import annotations

"""USearch-backed vector index with SQLite chunk metadata.

Stores chunk metadata in `chunks.sqlite` and vectors in `chunks.usearch`
next to the existing scan state database in the app data dir.

Comments are in English as requested.
"""

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from scan_state import get_app_dirs, APP_NAME
from fs_iter import TextChunk

# Support multiple USearch Python API layouts
UIndex = None  # type: ignore
UModule = None  # type: ignore
try:  # Preferred modern import path
    from usearch.index import Index as UIndex  # type: ignore
    import usearch as UModule  # type: ignore
except Exception:  # pragma: no cover
    try:
        import usearch as UModule  # type: ignore
        # Some versions expose Index at top-level
        UIndex = getattr(UModule, "Index", None)
    except Exception:
        UModule = None  # type: ignore


INDEX_FILE = "chunks.usearch"
CHUNKS_DB = "chunks.sqlite"


@dataclass
class IndexConfig:
    metric: str = "cos"
    dtype: str = "f32"
    connectivity: int = 16
    expansion_add: int = 128
    expansion_search: int = 64


class ChunkStore:
    """SQLite store for chunk metadata and mapping ids to files."""

    def __init__(self, db_path: Optional[Path] = None, *, app_name: str = APP_NAME):
        if db_path is None:
            dirs = get_app_dirs(app_name)
            db_path = dirs.data_dir / CHUNKS_DB
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA busy_timeout=5000;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                mtime REAL NOT NULL,
                mtime_iso TEXT NOT NULL,
                ver TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);
            CREATE INDEX IF NOT EXISTS chunks_deleted_idx ON chunks(deleted);
            """
        )
        cur.close()
        self.conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self.conn.commit()

    def allocate_chunk_ids(self, n: int) -> List[int]:
        cur = self.conn.execute("SELECT COALESCE(MAX(id), 0) FROM chunks")
        start = int(cur.fetchone()[0]) + 1
        return list(range(start, start + n))

    def add_chunks(self, ids: Sequence[int], chunks: Sequence[TextChunk], ver: str, *, deleted: int = 0) -> None:
        assert len(ids) == len(chunks)
        self.conn.executemany(
            """
            INSERT INTO chunks(id, path, chunk_index, start_char, end_char, mtime, mtime_iso, ver, deleted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    cid,
                    c.path,
                    c.chunk_index,
                    c.start_char,
                    c.end_char,
                    c.mtime,
                    c.mtime_iso,
                    ver,
                    int(deleted),
                )
                for cid, c in zip(ids, chunks)
            ],
        )
        self.conn.commit()

    def mark_chunks_active(self, ids: Sequence[int]) -> None:
        if not ids:
            return
        qmarks = ",".join(["?"] * len(ids))
        self.conn.execute(f"UPDATE chunks SET deleted=0 WHERE id IN ({qmarks})", list(ids))
        self.conn.commit()

    def remove_chunks_by_path(self, path: str) -> List[int]:
        """Delete rows for a file path and return their ids."""

        abspath = os.path.abspath(path)
        cur = self.conn.execute("SELECT id FROM chunks WHERE path=?", (abspath,))
        ids = [int(r[0]) for r in cur.fetchall()]
        if ids:
            self.conn.execute("DELETE FROM chunks WHERE path=?", (abspath,))
            self.conn.commit()
        return ids

    def remove_chunks_by_prefix(self, prefix: str) -> List[int]:
        """Delete rows for all files whose path starts with prefix; return ids.

        Note: callers should also remove ids from the USearch index.
        """

        absprefix = os.path.abspath(prefix)
        cur = self.conn.execute(
            "SELECT id FROM chunks WHERE path LIKE ? ESCAPE '\\'",
            (absprefix.replace("_", "\\_").replace("%", "\\%") + "%",),
        )
        ids = [int(r[0]) for r in cur.fetchall()]
        self.conn.execute(
            "DELETE FROM chunks WHERE path LIKE ? ESCAPE '\\'",
            (absprefix.replace("_", "\\_").replace("%", "\\%") + "%",),
        )
        self.conn.commit()
        return ids

    def get_chunk(self, cid: int) -> Optional[sqlite3.Row]:
        cur = self.conn.execute("SELECT * FROM chunks WHERE id=? AND deleted=0", (cid,))
        return cur.fetchone()

    def get_index_paths(self) -> Tuple[Path, Path]:
        base = self.db_path.parent
        return (base / INDEX_FILE, self.db_path)


class UsearchIndex:
    """Thin wrapper around usearch.Index with persistence and dimension checks."""

    def __init__(self, *, app_name: str = APP_NAME, cfg: Optional[IndexConfig] = None):
        if UIndex is None or UModule is None:
            raise RuntimeError("usearch is not installed. Please `pip install usearch`.")
        self.cfg = cfg or IndexConfig()
        dirs = get_app_dirs(app_name)
        self.index_path = dirs.data_dir / INDEX_FILE
        self.store = ChunkStore(dirs.data_dir / CHUNKS_DB, app_name=app_name)
        self.index: Optional["UIndex"] = None  # type: ignore[name-defined]

    # ---- dtype helpers ----
    @staticmethod
    def _np_dtype(dtype: str):
        import numpy as np  # local import to avoid hard dependency at import time

        m = {
            "f16": np.float16,
            "f32": np.float32,
            "f64": np.float64,
            "i8": np.int8,
            "u8": np.uint8,
        }
        return m.get(dtype, np.float32)

    @staticmethod
    def _normalize_rows(arr):
        import numpy as np
        a = np.asarray(arr)
        if a.ndim == 1:
            norm = np.linalg.norm(a)
            return a if norm == 0 else a / (norm + 1e-12)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return a / norms

    def close(self) -> None:
        self.index = None
        self.store.close()

    def ensure_open(self, dim: int) -> None:
        meta_dim = self.store.get_meta("dim")
        if self.index is not None:
            # Verify dimension matches
            if meta_dim and int(meta_dim) != dim:
                raise RuntimeError(f"Index dimension mismatch: existing={meta_dim}, new={dim}")
            return

        if self.index_path.exists():
            # Try to restore existing index from disk
            self.index = self._restore(self.index_path, expected_dim=int(meta_dim) if meta_dim else dim)
            # Check dim meta
            if meta_dim and int(meta_dim) != dim:
                raise RuntimeError(f"Index dimension mismatch: existing={meta_dim}, new={dim}")
        else:
            # Create new
            self.index = self._new_index(dim)
            # Persist empty index so it exists on disk after first save
            self.index.save(str(self.index_path))
            self.store.set_meta("dim", str(dim))
            self.store.set_meta("metric", self.cfg.metric)
            self.store.set_meta("dtype", self.cfg.dtype)

    def _new_index(self, dim: int):
        # Try full-parameter constructor; fall back to minimal signature
        try:
            return UIndex(
                metric=self.cfg.metric,
                dtype=self.cfg.dtype,
                ndim=dim,
                connectivity=self.cfg.connectivity,
                expansion_add=self.cfg.expansion_add,
                expansion_search=self.cfg.expansion_search,
            )
        except Exception:
            return UIndex(ndim=dim, metric=self.cfg.metric, dtype=self.cfg.dtype)  # type: ignore[call-arg]

    def _restore(self, path: Path, expected_dim: int):
        # Try static/class restore
        if hasattr(UIndex, "restore"):
            return UIndex.restore(str(path))  # type: ignore[attr-defined]
        # Try static/class load
        if hasattr(UIndex, "load"):
            try:
                return UIndex.load(str(path))  # type: ignore[attr-defined]
            except TypeError:
                # Maybe load is an instance method requiring a pre-created index
                idx = self._new_index(expected_dim)
                idx.load(str(path))  # type: ignore[attr-defined]
                return idx
        # No supported loader found
        raise RuntimeError("Unsupported usearch version: cannot restore index")

    def save(self) -> None:
        if self.index is not None:
            self.index.save(str(self.index_path))

    def add(self, ids: Sequence[int], vectors: Sequence[Sequence[float]]) -> None:
        if self.index is None:
            raise RuntimeError("Index not open")
        # Ensure NumPy arrays of correct dtypes
        import numpy as np

        keys = np.asarray(list(ids), dtype=np.uint64)
        vec_dtype = self._np_dtype(self.cfg.dtype)
        vecs = np.asarray(list(vectors), dtype=vec_dtype)
        # For cosine, normalize vectors to unit length to improve quality
        if self.cfg.metric.lower().startswith("cos"):
            vecs = self._normalize_rows(vecs).astype(vec_dtype, copy=False)
        try:
            self.index.add(keys, vecs)  # type: ignore[attr-defined]
        except Exception:
            # Some versions expect keyword names
            self.index.add(keys=keys, vectors=vecs)  # type: ignore[attr-defined]
        self.save()

    def remove(self, ids: Sequence[int]) -> None:
        if self.index is None:
            raise RuntimeError("Index not open")
        if not ids:
            return
        import numpy as np
        keys = np.asarray(list(ids), dtype=np.uint64)
        if hasattr(self.index, "remove"):
            try:
                self.index.remove(keys)  # type: ignore[attr-defined]
                self.save()
            except Exception:
                # As a fallback, we cannot remove; higher-level should ignore via DB.
                pass
        elif hasattr(self.index, "erase"):
            try:
                self.index.erase(keys)  # type: ignore[attr-defined]
                self.save()
            except Exception:
                pass

    def search(self, query: Sequence[float], k: int) -> Tuple[List[int], List[float]]:
        if self.index is None:
            raise RuntimeError("Index not open")
        import numpy as np
        vec_dtype = self._np_dtype(self.cfg.dtype)
        q = np.asarray(query, dtype=vec_dtype)
        if self.cfg.metric.lower().startswith("cos"):
            q = self._normalize_rows(q).astype(vec_dtype, copy=False)
        # Try search, fallback to knn
        if hasattr(self.index, "search"):
            result = self.index.search(q, k)  # type: ignore[attr-defined]
        else:
            result = self.index.knn(q, k)  # type: ignore[attr-defined]
        # Support both (ids, dists) or object with attributes
        if isinstance(result, tuple) and len(result) == 2:
            ids, dists = result
        else:
            ids = getattr(result, "keys", list())
            dists = getattr(result, "distances", list())
        return list(map(int, ids)), list(map(float, dists))
