esir — local semantic search with GGUF embeddings, USearch index, and a Tkinter GUI.

Overview
- Tk GUI: `tk_search_gui.py` — search page with query box, Top‑K, result list with collapsible snippets, and system open on click. Buttons to pick model and directory, and to run indexing.
- Indexing: streams text files, chunks, embeds with a GGUF, stores vectors in USearch (`chunks.usearch`) and chunk metadata in SQLite (`chunks.sqlite`). Incremental updates with safe “pending -> activate” flow and deleted-file cleanup.
- Scan state DB: `scan_state.sqlite` tracks file size/mtime and scanner version per file.

Quick start
- Install uv: https://docs.astral.sh/uv/getting-started/ (if not already installed)
- Create venv and sync deps: `uv venv && . .venv/bin/activate && uv sync`
- Optional nice themes: `uv sync --extra gui` (installs `ttkbootstrap`)
- Run GUI: `python3 tk_search_gui.py` or `esir-gui`
  - Model: click “Browse…” to pick a `.gguf`, or put your model under `./models/` (auto‑detected on first run).
  - Directory: pick a folder with small text files.
  - Index: builds/updates the USearch index; only changed files are reprocessed.
  - Search: type your query; tune `Top‑K` (default 100).

Settings and data locations
- INI: `settings.ini` in the platform config dir (via XDG/AppData/Library; see `scan_state.get_app_dirs`).
- Data: USearch/SQLite live in the app data dir chosen by `get_app_dirs()`.
- Environment override: `ESIR_MODEL_PATH=/path/to/model.gguf` forces the default model.

Bundle a default model
- Put your standard `.gguf` under `models/` in the project root. The GUI will auto‑pick it on first run if `ESIR_MODEL_PATH` isn’t set. The model is loaded only when you index/search.

Notes
- USearch compatibility: the wrapper handles multiple API layouts (`usearch.index.Index` vs `usearch.Index`) and expects NumPy arrays.
- Cosine metric: vectors are L2-normalized on add/search; GUI shows similarity (1 − distance).
- Bi-encoder prompts: if the model filename matches bge/e5/gte, documents are embedded with “passage: …” and queries with “query: …”.
