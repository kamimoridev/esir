from __future__ import annotations

"""Library API for indexing and searching without any GUI dependencies.

Provides simple functions:
 - index_directory(): incremental indexing with pending/activation and deleted-file cleanup
 - search_query(): embed a query and return ranked chunks with metadata
 - clear_all(): remove entire index and metadata
 - clear_under_root(): remove index entries under a directory prefix

Comments are in English as requested.
"""

import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from embeddings import GGUFEmbedder, ModelConfig
from fs_iter import TextChunk, filter_text_files, generate_chunks, iter_files, _open_text_with_best_effort
from scan_state import ScanState
from vector_index_usearch import UsearchIndex


def _format_doc_text(model_path: str, text: str) -> str:
    name = os.path.basename(model_path).lower()
    if any(x in name for x in ("bge", "e5", "gte")):
        return f"passage: {text}"
    return text


def _format_query_text(model_path: str, text: str) -> str:
    name = os.path.basename(model_path).lower()
    if any(x in name for x in ("bge", "e5", "gte")):
        return f"query: {text}"
    return text


def index_directory(
    model_path: str,
    root: str,
    *,
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
    scanner_ver: str = "1",
    batch_size: int = 32,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Incrementally index a directory. Returns (files_indexed, chunks_added)."""

    def emit(msg: str) -> None:
        if log:
            log(msg)

    embedder = GGUFEmbedder()
    embedder.load(ModelConfig(model_path=model_path))
    dim = len(embedder.embed_text("esir init"))
    index = UsearchIndex()
    index.ensure_open(dim)
    state = ScanState()

    files_indexed = 0
    chunks_added = 0

    items = filter_text_files(iter_files(root, include_metadata=True))
    changed_infos: List = []
    current_paths = set()
    for fi in items:
        current_paths.add(fi.path)
        if state.has_changed(fi, ver=scanner_ver):
            changed_infos.append(fi)

    emit(f"Changed files: {len(changed_infos)}")

    # Remove deleted files under this root
    removed = 0
    root_prefix = os.path.abspath(root) + os.sep
    for path, _size, _mtime, _ver in state.iter_all():
        ap = os.path.abspath(path)
        if not ap.startswith(root_prefix):
            continue
        if ap not in current_paths and not os.path.exists(ap):
            ids = index.store.remove_chunks_by_path(ap)
            try:
                index.remove(ids)
            except Exception:
                pass
            state.remove_path(ap)
            removed += 1
    if removed:
        emit(f"Removed {removed} missing files from index")

    if not changed_infos:
        emit("No changes detected")
        return (0, 0)

    for fi in changed_infos:
        path = fi.path
        emit(f"Indexing {os.path.basename(path)} …")
        # Remove previous chunks
        ids_prev = index.store.remove_chunks_by_path(path)
        if ids_prev:
            try:
                index.remove(ids_prev)
            except Exception:
                pass
        # Mark pending
        state.upsert_file(fi, ver=f"{scanner_ver}:pending")
        # Generate chunks
        chunks: List[TextChunk] = list(
            generate_chunks(path, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        )
        if not chunks:
            state.upsert_file(fi, ver=scanner_ver)
            continue
        ids = index.store.allocate_chunk_ids(len(chunks))
        # Stage rows as deleted=1
        index.store.add_chunks(ids, chunks, ver=scanner_ver, deleted=1)
        # Embed in batches
        texts = [_format_doc_text(model_path, c.text) for c in chunks]
        vectors: List[List[float]] = []
        for s in range(0, len(texts), batch_size):
            batch = texts[s : s + batch_size]
            vectors.extend(embedder.embed_texts(batch))
        # Add to index
        index.add(ids, vectors)
        # Activate and finalize
        index.store.mark_chunks_active(ids)
        state.upsert_file(fi, ver=scanner_ver)
        files_indexed += 1
        chunks_added += len(chunks)

    emit("Indexing complete")
    return (files_indexed, chunks_added)


def search_query(model_path: str, query: str, *, k: int = 100) -> List[Dict]:
    """Search the USearch index using the embedding model. Returns list of hits.

    Each hit: { id, text, chunk_index, score, meta: { path, mtime_iso, start, end } }.
    """

    embedder = GGUFEmbedder()
    embedder.load(ModelConfig(model_path=model_path))
    qtext = _format_query_text(model_path, query)
    qvec = embedder.embed_text(qtext)
    index = UsearchIndex()
    index.ensure_open(len(qvec))
    ids, dists = index.search(qvec, k)
    metric = (index.store.get_meta("metric") or "cos").lower()

    def to_score(distance: float) -> float:
        if metric.startswith("cos"):
            sim = 1.0 - float(distance)
            return max(0.0, min(1.0, sim))
        return 1.0 / (1.0 + float(distance))

    rows: List[Dict] = []
    for cid, dist in zip(ids, dists):
        row = index.store.get_chunk(cid)
        if row is None:
            continue
        path = row["path"]
        start = int(row["start_char"]) if row["start_char"] is not None else 0
        end = int(row["end_char"]) if row["end_char"] is not None else start
        try:
            text = _open_text_with_best_effort(path)
            snippet = text[start:end]
        except Exception:
            snippet = "<unavailable>"
        rows.append(
            {
                "id": int(cid),
                "text": snippet,
                "chunk_index": int(row["chunk_index"]),
                "score": float(to_score(dist)),
                "meta": {
                    "path": path,
                    "mtime_iso": row["mtime_iso"],
                    "start": start,
                    "end": end,
                },
            }
        )
    return rows


def clear_all() -> None:
    """Remove the entire index and chunk store (and scan state)."""

    from scan_state import get_app_dirs
    from vector_index_usearch import INDEX_FILE, CHUNKS_DB

    dirs = get_app_dirs()
    # Delete files if exist
    for path in (dirs.data_dir / INDEX_FILE, dirs.data_dir / CHUNKS_DB, dirs.data_dir / "scan_state.sqlite"):
        try:
            if Path(path).exists():
                Path(path).unlink()
        except Exception:
            pass


def clear_under_root(root: str) -> int:
    """Remove all chunks and scan state entries under a directory prefix.

    Returns number of files removed from scan state.
    """

    index = UsearchIndex()
    store = index.store
    # Remove chunks by exact path for all files under root using scan state listing
    state = ScanState()
    root_prefix = os.path.abspath(root) + os.sep
    removed = 0
    for path, _size, _mtime, _ver in list(state.iter_all()):
        ap = os.path.abspath(path)
        if ap.startswith(root_prefix):
            ids = store.remove_chunks_by_path(ap)
            try:
                index.remove(ids)
            except Exception:
                pass
            state.remove_path(ap)
            removed += 1
    return removed

