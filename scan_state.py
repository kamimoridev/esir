from __future__ import annotations

"""Cross-platform scan state with SQLite.

Stores per-file size, mtime, and a scanner version. Supports efficient
change checks on subsequent scans.

Comments are in English as requested.
"""

import os
import sys
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple, Union

from fs_iter import FileInfo, _ensure_fileinfo


APP_NAME = "esir"


@dataclass(frozen=True)
class AppDirs:
    config_dir: Path
    data_dir: Path
    cache_dir: Path


def get_app_dirs(app_name: str = APP_NAME, *, create: bool = True) -> AppDirs:
    """Return cross-platform config/data/cache dirs, creating them if needed.

    - Linux (XDG):
        config: $XDG_CONFIG_HOME/<app>/ or ~/.config/<app>
        data:   $XDG_STATE_HOME/<app>/ or ~/.local/state/<app> (fallback to $XDG_DATA_HOME/<app>)
        cache:  $XDG_CACHE_HOME/<app>/ or ~/.cache/<app>
    - macOS:
        config+data: ~/Library/Application Support/<app>
        cache:       ~/Library/Caches/<app>
    - Windows:
        config+data: %APPDATA%\<app>
        cache:       %LOCALAPPDATA%\<app>\Cache
    """

    home = Path.home()
    if os.name == "nt":
        appdata = Path(os.getenv("APPDATA", home / "AppData" / "Roaming"))
        local = Path(os.getenv("LOCALAPPDATA", home / "AppData" / "Local"))
        config_dir = appdata / app_name
        data_dir = appdata / app_name
        cache_dir = local / app_name / "Cache"
    elif sys.platform == "darwin":  # type: ignore[name-defined]
        base = home / "Library"
        config_dir = base / "Application Support" / app_name
        data_dir = base / "Application Support" / app_name
        cache_dir = base / "Caches" / app_name
    else:
        xdg_config = Path(os.getenv("XDG_CONFIG_HOME", home / ".config"))
        xdg_state = Path(os.getenv("XDG_STATE_HOME", home / ".local" / "state"))
        xdg_data = Path(os.getenv("XDG_DATA_HOME", home / ".local" / "share"))
        xdg_cache = Path(os.getenv("XDG_CACHE_HOME", home / ".cache"))
        config_dir = xdg_config / app_name
        # prefer state; fallback to data
        data_dir = (xdg_state if xdg_state else xdg_data) / app_name
        cache_dir = xdg_cache / app_name

    if create:
        for d in (config_dir, data_dir, cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    return AppDirs(config_dir=config_dir, data_dir=data_dir, cache_dir=cache_dir)


class ScanState:
    """SQLite-backed storage for scanned file info and scanner version."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None, *, app_name: str = APP_NAME):
        if db_path is None:
            dirs = get_app_dirs(app_name)
            db_path = dirs.data_dir / "scan_state.sqlite"
        self.db_path = Path(db_path)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ------------- Schema -------------
    def _init_schema(self) -> None:
        cur = self.conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                size INTEGER NOT NULL,
                mtime REAL NOT NULL,
                ver TEXT NOT NULL
            );
            """
        )
        cur.close()
        self.conn.commit()

    # ------------- Meta -------------
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

    # ------------- Files -------------
    def has_changed(self, item: Union[str, FileInfo], *, ver: str, follow_symlinks: bool = False) -> bool:
        """Return True if file is new, size/mtime changed, or version differs."""

        info = _ensure_fileinfo(item, follow_symlinks=follow_symlinks)
        if info is None:
            return False
        cur = self.conn.execute("SELECT size, mtime, ver FROM files WHERE path=?", (info.path,))
        row = cur.fetchone()
        if row is None:
            return True
        size_changed = int(row[0]) != info.size
        mtime_changed = abs(float(row[1]) - info.mtime) > 1e-6
        ver_changed = (row[2] or "") != ver
        return size_changed or mtime_changed or ver_changed

    def upsert_file(self, item: Union[str, FileInfo], *, ver: str, follow_symlinks: bool = False) -> Optional[FileInfo]:
        """Insert or update file row with current size/mtime/version."""

        info = _ensure_fileinfo(item, follow_symlinks=follow_symlinks)
        if info is None:
            return None
        self.conn.execute(
            """
            INSERT INTO files(path, size, mtime, ver)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                size=excluded.size,
                mtime=excluded.mtime,
                ver=excluded.ver
            """,
            (info.path, info.size, info.mtime, ver),
        )
        self.conn.commit()
        return info

    def remove_path(self, path: Union[str, Path]) -> None:
        self.conn.execute("DELETE FROM files WHERE path=?", (os.path.abspath(str(path)),))
        self.conn.commit()

    def iter_all(self) -> Iterator[Tuple[str, int, float, str]]:
        """Yield (path, size, mtime, ver) for all entries."""

        cur = self.conn.execute("SELECT path, size, mtime, ver FROM files")
        for row in cur:
            yield (row[0], int(row[1]), float(row[2]), row[3])


def filter_changed(
    items: Iterable[Union[str, FileInfo]],
    state: ScanState,
    *,
    ver: str,
    follow_symlinks: bool = False,
) -> Iterator[FileInfo]:
    """Stream only the items that are considered changed per ScanState.

    Typical usage:
        state = ScanState()
        for fi in filter_changed(iter_files(root, include_metadata=True), state, ver="1"):
            ... process ...
            state.upsert_file(fi, ver="1")
    """

    for item in items:
        if state.has_changed(item, ver=ver, follow_symlinks=follow_symlinks):
            info = _ensure_fileinfo(item, follow_symlinks=follow_symlinks)
            if info is not None:
                yield info
