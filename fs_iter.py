from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator, Iterable, Iterator, Optional, Union


@dataclass(frozen=True)
class FileInfo:
    """Lightweight file metadata for a single filesystem entry.

    Fields are chosen to be broadly useful and cheap to fetch.
    Times are provided as both raw POSIX timestamp and ISO string for convenience.
    """

    path: str
    size: int
    mtime: float
    mtime_iso: str
    is_symlink: bool


def iter_files(
    root: Union[str, Path],
    *,
    include_metadata: bool = True,
    follow_symlinks: bool = False,
    on_error: Optional[Callable[[Exception, str], None]] = None,
) -> Iterator[Union[str, FileInfo]]:
    """Yield files under `root` recursively, optionally with metadata.

    This function is memory-efficient: it streams results using os.scandir and
    does not accumulate paths in memory. It avoids recursion depth limits by
    using an explicit stack.

    Args:
        root: Directory to scan.
        include_metadata: When True, yields FileInfo; otherwise yields str paths.
        follow_symlinks: Whether to follow directory symlinks while descending.
        on_error: Optional callback called as (exc, path) when a directory
                  cannot be read or a file's stat fails. By default errors are ignored.

    Yields:
        Either `FileInfo` or `str` (absolute path), depending on `include_metadata`.
    """

    root_path = Path(root)
    # Always work with absolute paths so results are unambiguous.
    try:
        start_dir = root_path.resolve(strict=True)
    except Exception:
        # If the starting point cannot be resolved strictly, fall back to absolute.
        start_dir = root_path.absolute()

    stack: list[Path] = [start_dir]

    while stack:
        current_dir = stack.pop()
        try:
            with os.scandir(current_dir) as it:
                for entry in it:
                    # Skip broken symlinks to directories unless following symlinks.
                    try:
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            stack.append(Path(entry.path))
                            continue
                    except FileNotFoundError:
                        # Entry may have disappeared; optionally report and continue.
                        if on_error is not None:
                            on_error(FileNotFoundError(entry.path), entry.path)
                        continue

                    # Process files and other non-directory entries.
                    if include_metadata:
                        try:
                            st = entry.stat(follow_symlinks=follow_symlinks)
                            mtime = float(st.st_mtime)
                            mtime_iso = datetime.fromtimestamp(mtime).isoformat()
                            yield FileInfo(
                                path=os.path.abspath(entry.path),
                                size=int(st.st_size),
                                mtime=mtime,
                                mtime_iso=mtime_iso,
                                is_symlink=entry.is_symlink(),
                            )
                        except Exception as exc:
                            if on_error is not None:
                                on_error(exc, entry.path)
                            # Skip entries that cannot be stat'ed.
                            continue
                    else:
                        yield os.path.abspath(entry.path)
        except Exception as exc:
            # Handle unreadable directories gracefully.
            if on_error is not None:
                on_error(exc, str(current_dir))
            continue


def iter_paths(root: Union[str, Path], *, follow_symlinks: bool = False) -> Iterator[str]:
    """Convenience wrapper that yields only absolute paths.

    Equivalent to `iter_files(root, include_metadata=False, follow_symlinks=...)`.
    """

    yield from iter_files(root, include_metadata=False, follow_symlinks=follow_symlinks)


# -------------------------
# Text filtering utilities
# -------------------------

# Common text-like extensions. This is a heuristic; content sniffing is also used.
TEXT_EXTS = {
    ".txt",
    ".md",
    ".rst",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".csv",
    ".tsv",
    ".log",
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".xml",
    ".sql",
    ".sh",
    ".bat",
    ".ps1",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".h",
    ".cpp",
    ".hpp",
    ".cc",
    ".hh",
    ".m",
    ".mm",
    ".rb",
    ".php",
    ".pl",
    ".lua",
    ".swift",
    ".scala",
    ".gradle",
    ".dockerfile",
    ".make",
    ".mk",
}


def _looks_text_by_extension(path: str) -> bool:
    """Quick extension-based guess if file is text."""

    ext = Path(path).suffix.lower()
    if ext:
        return ext in TEXT_EXTS
    # Also consider files without extension but with known names as text-like.
    name = Path(path).name.lower()
    known = {"dockerfile", "makefile", "license", "readme"}
    return name in known


def _looks_text_by_sniff(path: str, sniff_bytes: int = 8192) -> bool:
    """Heuristically detect if file looks like text by inspecting bytes.

    Considers presence of NUL bytes as binary. Computes ratio of bytes outside
    a conservative printable set; if too high, treats as binary.
    """

    try:
        with open(path, "rb") as f:
            sample = f.read(sniff_bytes)
    except Exception:
        return False

    if b"\x00" in sample:
        return False

    # Allow common whitespace and printable ASCII; tolerate high-bit bytes.
    text_bytes = (
        b"\t\n\r\x0b\x0c"  # whitespace
        + bytes(range(32, 127))  # printable ASCII
    )
    nontext = sum(1 for b in sample if b not in text_bytes)
    # If more than 30% non-text-like bytes, assume binary.
    return (len(sample) == 0) or (nontext / max(1, len(sample)) <= 0.30)


def _ensure_fileinfo(
    item: Union[str, FileInfo], *, follow_symlinks: bool = False
) -> Optional[FileInfo]:
    """Normalize an item to FileInfo, or return None if stat fails."""

    if isinstance(item, FileInfo):
        return item
    try:
        st = os.stat(item, follow_symlinks=follow_symlinks)
        mtime = float(st.st_mtime)
        return FileInfo(
            path=os.path.abspath(str(item)),
            size=int(st.st_size),
            mtime=mtime,
            mtime_iso=datetime.fromtimestamp(mtime).isoformat(),
            is_symlink=os.path.islink(item),
        )
    except Exception:
        return None


def filter_text_files(
    files: Iterable[Union[str, FileInfo]],
    *,
    max_size_bytes: int = 10 * 1024 * 1024,
    sniff_bytes: int = 8192,
    follow_symlinks: bool = False,
) -> Iterator[FileInfo]:
    """Filter an incoming file iterator to only text files under size limit.

    Accepts either `str` paths or `FileInfo` and yields `FileInfo` only.
    Uses a combination of extension and content sniffing to classify text files.
    Files larger than `max_size_bytes` are excluded.
    """

    for item in files:
        info = _ensure_fileinfo(item, follow_symlinks=follow_symlinks)
        if info is None:
            continue
        if info.size > max_size_bytes:
            continue
        # First check by extension, then confirm with content sniff to avoid
        # misclassifying binaries with text-like extensions.
        if _looks_text_by_extension(info.path) or _looks_text_by_sniff(
            info.path, sniff_bytes=sniff_bytes
        ):
            yield info


# -------------------------
# Chunking for embeddings
# -------------------------

@dataclass(frozen=True)
class TextChunk:
    """A single text chunk with embedding-friendly metadata."""

    path: str
    size: int
    mtime: float
    mtime_iso: str
    chunk_index: int
    start_char: int
    end_char: int
    overlap_chars: int
    text: str


def _open_text_with_best_effort(path: str) -> str:
    """Read file as text with a pragmatic decoding strategy.

    - Detect UTF BOMs (UTF-8/UTF-16 LE/BE) via Python's 'utf-8-sig' etc.
    - Try utf-8; on failure, read as bytes and decode with 'utf-8' using
      replacement to avoid exceptions.
    This keeps implementation lightweight and dependency-free.
    """

    # First try UTF variants that handle BOMs transparently.
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        pass
    # Try UTF-16 with BOM auto-detection.
    try:
        with open(path, "r", encoding="utf-16") as f:
            return f.read()
    except UnicodeError:
        pass
    # Fallback: read bytes and replace undecodable sequences.
    with open(path, "rb") as f:
        data = f.read()
    return data.decode("utf-8", errors="replace")


def generate_chunks(
    file: Union[str, FileInfo],
    *,
    chunk_chars: int = 2000,
    overlap_chars: int = 200,
    follow_symlinks: bool = False,
) -> Iterator[TextChunk]:
    """Generate embedding-ready chunks from a single file.

    Reads the file into memory as text (bounded by size filter) and slices
    into overlapping windows. Yields `TextChunk` objects with rich metadata.

    Args:
        file: A path or FileInfo for a text file.
        chunk_chars: Target maximum characters per chunk.
        overlap_chars: Characters to overlap between consecutive chunks.
        follow_symlinks: Whether to stat symlinks if `file` is a path.

    Notes:
        - Defaults are tuned for modern embedding models: ~2000 chars per
          chunk (~800–1200 tokens) with ~10% overlap for context carryover.
        - For code-heavy corpora, smaller chunks (800–1200 chars) can help.
    """

    assert chunk_chars > 0, "chunk_chars must be positive"
    assert 0 <= overlap_chars < chunk_chars, "overlap_chars must be in [0, chunk_chars)"

    info = _ensure_fileinfo(file, follow_symlinks=follow_symlinks)
    if info is None:
        return iter(())  # type: ignore[return-value]

    text = _open_text_with_best_effort(info.path)
    n = len(text)
    if n == 0:
        return iter(())  # type: ignore[return-value]

    step = chunk_chars - overlap_chars
    index = 0
    pos = 0
    while pos < n:
        end = min(n, pos + chunk_chars)
        chunk_text = text[pos:end]
        yield TextChunk(
            path=info.path,
            size=info.size,
            mtime=info.mtime,
            mtime_iso=info.mtime_iso,
            chunk_index=index,
            start_char=pos,
            end_char=end,
            overlap_chars=overlap_chars,
            text=chunk_text,
        )
        index += 1
        pos += step

