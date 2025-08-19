from __future__ import annotations

"""Console CLI for indexing and searching without GUI.

Commands:
  esir index  -m MODEL -r ROOT [--chunk-chars N] [--overlap-chars N] [--batch-size N] [--ver V]
  esir search -m MODEL -q QUERY [-k N]
  esir clear  [--all | --root DIR]

Comments are in English as requested.
"""

import argparse
import json
import sys
from typing import Any

from esir_lib import clear_all, clear_under_root, index_directory, search_query


def _add_common_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", required=True, help="Path to GGUF embedding model")


def cmd_index(args: argparse.Namespace) -> int:
    def log(msg: str) -> None:
        print(msg, file=sys.stderr)

    files, chunks = index_directory(
        model_path=args.model,
        root=args.root,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
        scanner_ver=args.ver,
        batch_size=args.batch_size,
        log=log,
    )
    print(json.dumps({"files_indexed": files, "chunks_added": chunks}))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    rows = search_query(args.model, args.query, k=args.k)
    if args.json:
        print(json.dumps(rows, ensure_ascii=False))
    else:
        for r in rows:
            meta = r.get("meta", {})
            print(f"{r.get('score', 0):.3f} {meta.get('path','')}#{r.get('chunk_index')}\n{r.get('text','')[:2000]}\n")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    if args.all:
        clear_all()
        print("{\"cleared\": \"all\"}")
        return 0
    if args.root:
        n = clear_under_root(args.root)
        print(json.dumps({"cleared_under_root": n}))
        return 0
    print("Nothing to clear. Use --all or --root DIR", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="esir", description="Local semantic indexer & search (GGUF + USearch)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Index a directory incrementally")
    _add_common_model(p_index)
    p_index.add_argument("-r", "--root", required=True, help="Directory to scan")
    p_index.add_argument("--chunk-chars", type=int, default=2000, help="Chunk size in characters")
    p_index.add_argument("--overlap-chars", type=int, default=200, help="Overlap between chunks")
    p_index.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p_index.add_argument("--ver", default="1", help="Scanner version string")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search the index with a query")
    _add_common_model(p_search)
    p_search.add_argument("-q", "--query", required=True, help="Search query text")
    p_search.add_argument("-k", type=int, default=100, help="Top K results")
    p_search.add_argument("--json", action="store_true", help="Output JSON")
    p_search.set_defaults(func=cmd_search)

    p_clear = sub.add_parser("clear", help="Clear index and metadata")
    g = p_clear.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="Clear entire index and metadata")
    g.add_argument("--root", help="Clear only entries under directory prefix")
    p_clear.set_defaults(func=cmd_clear)

    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


def main_index() -> int:
    return main(["index", *sys.argv[1:]])


def main_search() -> int:
    return main(["search", *sys.argv[1:]])


def main_clear() -> int:
    return main(["clear", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())

