from __future__ import annotations

"""Tkinter-based search GUI using esir_lib and INI settings.

Features:
- Pick model (.gguf) and directory; settings saved to INI via get_app_dirs().
- Index incrementally using esir_lib.index_directory (runs in a worker thread).
- Search using esir_lib.search_query with Top-K control.
- Clean, responsive UI using ttk with a simple light theme and result cards.
- Open files with the system default application on click.

Comments are in English as requested.
"""

import configparser
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

# Optional nicer themes via ttkbootstrap
try:
    import ttkbootstrap as tb  # type: ignore
except Exception:  # pragma: no cover - optional
    tb = None  # type: ignore

from scan_state import get_app_dirs
from esir_lib import index_directory, search_query


@dataclass
class Settings:
    model_path: str = ""
    root_path: str = ""
    chunk_chars: int = 2000
    overlap_chars: int = 200
    k: int = 100
    theme: str = ""  # ttkbootstrap theme name (e.g., 'flatly', 'darkly'); empty for default


def system_open(path: str) -> None:
    """Open a file with the default system handler."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except Exception:
        # Best-effort; ignore failures
        pass


class SettingsStore:
    """Load/save INI settings under the app config dir."""

    def __init__(self) -> None:
        self.dirs = get_app_dirs()
        self.path = self.dirs.config_dir / "settings.ini"

    def load(self) -> Settings:
        s = Settings()
        try:
            if self.path.exists():
                cp = configparser.ConfigParser()
                cp.read(self.path)
                sec = cp["esir"] if "esir" in cp else {}
                s.model_path = sec.get("model_path", "")
                s.root_path = sec.get("root_path", "")
                s.chunk_chars = int(sec.get("chunk_chars", 2000))
                s.overlap_chars = int(sec.get("overlap_chars", 200))
                s.k = int(sec.get("k", 100))
                s.theme = sec.get("theme", "")
        except Exception:
            pass
        # Auto-discover model if not set
        if not s.model_path or not Path(s.model_path).exists():
            cand = self._discover_default_model()
            if cand:
                s.model_path = str(cand)
                self.save(s)
        return s

    def save(self, s: Settings) -> None:
        cp = configparser.ConfigParser()
        cp["esir"] = {
            "model_path": s.model_path,
            "root_path": s.root_path,
            "chunk_chars": str(s.chunk_chars),
            "overlap_chars": str(s.overlap_chars),
            "k": str(s.k),
            "theme": s.theme or "",
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            cp.write(f)

    def _app_dir(self) -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys.executable).resolve().parent
        return Path(__file__).resolve().parent

    def _discover_default_model(self) -> Optional[Path]:
        env = os.getenv("ESIR_MODEL_PATH")
        if env and Path(env).exists():
            return Path(env)
        base = self._app_dir()
        for folder in (base / "models", base):
            if folder.exists():
                found = sorted(folder.glob("*.gguf"))
                if found:
                    return found[0]
        for p in Path.cwd().glob("*.gguf"):
            return p
        return None


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ESIR — Semantic Search")
        self.geometry("1100x780")
        self.minsize(800, 500)
        self._store = SettingsStore()
        self.settings = self._store.load()

        # Worker state
        self._index_thread: Optional[threading.Thread] = None
        self._search_thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[tuple[str, object]]" = queue.Queue()

        self._init_style()
        self._build_ui()
        self._poll_queue()

    # ---------- UI ----------
    def _init_style(self) -> None:
        # If ttkbootstrap is available, apply its theme
        if tb is not None:
            theme = self.settings.theme or ("darkly" if self._prefers_dark() else "flatly")
            try:
                self._tb_style = tb.Style(theme=theme)  # type: ignore[attr-defined]
            except Exception:
                self._tb_style = tb.Style()  # type: ignore[attr-defined]
        # Configure ttk styles regardless of theme backend
        style = ttk.Style(self)
        try:
            if tb is None:
                # Fall back to decent native themes
                if sys.platform.startswith("win"):
                    style.theme_use("vista")
                elif sys.platform == "darwin":
                    style.theme_use("aqua")
                else:
                    style.theme_use("clam")
        except Exception:
            pass
        style.configure("TButton", padding=6)
        style.configure("TEntry", padding=4)
        style.configure("Muted.TLabel", foreground="#666")
        style.configure("Title.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Path.TLabel", foreground="#0a6522")

    def _prefers_dark(self) -> bool:
        # Heuristic: respect desktop preference if available
        try:
            if sys.platform == "darwin":
                import subprocess
                out = subprocess.check_output([
                    "defaults", "read", "-g", "AppleInterfaceStyle"
                ], stderr=subprocess.DEVNULL)
                return bool(out.strip())
        except Exception:
            pass
        # On others, default to light
        return False

    def _build_ui(self) -> None:
        root = ttk.Frame(self)
        root.pack(fill=tk.BOTH, expand=True)

        # Top bar: query + controls
        top = ttk.Frame(root)
        top.pack(fill=tk.X, padx=10, pady=10)

        self.q_var = tk.StringVar()
        q_entry = ttk.Entry(top, textvariable=self.q_var)
        q_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
        q_entry.insert(0, "Type your query…")
        q_entry.bind("<Return>", lambda e: self.on_search())
        q_entry.bind("<FocusIn>", lambda e: self._clear_placeholder(q_entry))

        ttk.Label(top, text="Top-K:").pack(side=tk.LEFT)
        self.k_var = tk.IntVar(value=self.settings.k)
        k_spin = ttk.Spinbox(top, from_=1, to=10000, textvariable=self.k_var, width=6)
        k_spin.pack(side=tk.LEFT, padx=(4, 10))

        ttk.Button(top, text="Search", command=self.on_search).pack(side=tk.LEFT)

        # Options bar: model, dir, index
        bar = ttk.Frame(root)
        bar.pack(fill=tk.X, padx=10)

        self.model_var = tk.StringVar(value=self.settings.model_path)
        self.dir_var = tk.StringVar(value=self.settings.root_path)

        ttk.Label(bar, text="Model:").pack(side=tk.LEFT)
        self.model_entry = ttk.Entry(bar, textvariable=self.model_var, width=45)
        self.model_entry.pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(bar, text="Browse…", command=self.on_pick_model).pack(side=tk.LEFT)

        ttk.Label(bar, text="Directory:").pack(side=tk.LEFT, padx=(12, 0))
        self.dir_entry = ttk.Entry(bar, textvariable=self.dir_var, width=35)
        self.dir_entry.pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(bar, text="Choose…", command=self.on_pick_dir).pack(side=tk.LEFT)

        ttk.Button(bar, text="Index", command=self.on_index).pack(side=tk.LEFT, padx=(12, 0))

        # Theme selector if ttkbootstrap is available
        if tb is not None:
            ttk.Label(bar, text="Theme:").pack(side=tk.LEFT, padx=(12, 4))
            self.theme_var = tk.StringVar(value=self.settings.theme or ("darkly" if self._prefers_dark() else "flatly"))
            try:
                themes = sorted(set(self._tb_style.theme_names()))  # type: ignore[attr-defined]
            except Exception:
                themes = [self.theme_var.get()]
            self.theme_combo = ttk.Combobox(bar, textvariable=self.theme_var, values=themes, width=12, state="readonly")
            self.theme_combo.pack(side=tk.LEFT)
            self.theme_combo.bind("<<ComboboxSelected>>", lambda e: self.on_theme_change())

        # Status line
        self.status_var = tk.StringVar(value="Ready")
        status = ttk.Label(root, textvariable=self.status_var, style="Muted.TLabel")
        status.pack(fill=tk.X, padx=12, pady=(6, 4))

        # Results area: scrollable frame with result cards
        wrap = ttk.Frame(root)
        wrap.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.canvas = tk.Canvas(wrap, highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.cards = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.cards, anchor="nw")
        self.cards.bind("<Configure>", self._on_cards_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

    def _on_cards_configure(self, event):
        # Update scrollregion to encompass result cards
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Match inner frame width to canvas width
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def _clear_placeholder(self, entry: ttk.Entry) -> None:
        if entry.get().strip() == "Type your query…":
            entry.delete(0, tk.END)

    # ---------- Actions ----------
    def on_pick_model(self) -> None:
        path = filedialog.askopenfilename(title="Select GGUF model", filetypes=[("GGUF files", "*.gguf"), ("All files", "*.*")])
        if not path:
            return
        self.model_var.set(path)
        self.settings.model_path = path
        self._store.save(self.settings)
        self._set_status(f"Model set: {os.path.basename(path)}")

    def on_pick_dir(self) -> None:
        path = filedialog.askdirectory(title="Select directory")
        if not path:
            return
        self.dir_var.set(path)
        self.settings.root_path = path
        self._store.save(self.settings)
        self._set_status(f"Directory set: {path}")

    def on_index(self) -> None:
        if self._index_thread and self._index_thread.is_alive():
            messagebox.showinfo("Indexing", "Indexing already in progress.")
            return
        model = self.model_var.get().strip()
        root = self.dir_var.get().strip()
        if not model or not Path(model).exists():
            messagebox.showwarning("Model", "Please select a valid GGUF model.")
            return
        if not root or not Path(root).exists():
            messagebox.showwarning("Directory", "Please select a valid directory.")
            return
        # Persist settings in case user changed entries
        self.settings.model_path = model
        self.settings.root_path = root
        self._store.save(self.settings)
        self._set_status("Indexing…")

        def work():
            try:
                files_indexed, chunks_added = index_directory(
                    model,
                    root,
                    chunk_chars=self.settings.chunk_chars,
                    overlap_chars=self.settings.overlap_chars,
                    log=lambda m: self._queue.put(("log", m)),
                )
                self._queue.put(("log", f"Done. Files indexed: {files_indexed}, chunks added: {chunks_added}"))
            except Exception as exc:
                self._queue.put(("log", f"Index error: {exc}"))

        self._index_thread = threading.Thread(target=work, daemon=True)
        self._index_thread.start()

    def on_search(self) -> None:
        if self._search_thread and self._search_thread.is_alive():
            return
        model = self.model_var.get().strip()
        if not model:
            messagebox.showwarning("Model", "Please select a GGUF model first.")
            return
        q = self.q_var.get().strip()
        if not q or q == "Type your query…":
            return
        try:
            k = int(self.k_var.get())
        except Exception:
            k = 100
        k = max(1, min(10000, k))
        self.settings.k = k
        self.settings.model_path = model
        self._store.save(self.settings)
        self._set_status("Searching…")

        def work():
            try:
                rows = search_query(model, q, k=k)
                self._queue.put(("results", rows))
                self._queue.put(("log", f"Results: {len(rows)}"))
            except Exception as exc:
                self._queue.put(("results", []))
                self._queue.put(("log", f"Search error: {exc}"))

        self._search_thread = threading.Thread(target=work, daemon=True)
        self._search_thread.start()

    def on_theme_change(self) -> None:
        if tb is None:
            return
        name = (self.theme_var.get() if hasattr(self, "theme_var") else "").strip()
        if not name:
            return
        try:
            self._tb_style.theme_use(name)  # type: ignore[attr-defined]
            self.settings.theme = name
            self._store.save(self.settings)
            self._set_status(f"Theme: {name}")
        except Exception as exc:
            self._set_status(f"Theme error: {exc}")

    # ---------- Rendering ----------
    def _render_results(self, rows: List[dict]) -> None:
        for w in self.cards.winfo_children():
            w.destroy()
        if not rows:
            ttk.Label(self.cards, text="No results", style="Muted.TLabel").pack(anchor="w", padx=6, pady=6)
            return
        for r in rows:
            self._add_result_card(r)

    def _add_result_card(self, r: dict) -> None:
        frame = ttk.Frame(self.cards, relief=tk.GROOVE, padding=8)
        frame.pack(fill=tk.X, expand=True, padx=2, pady=6)

        path = r.get("meta", {}).get("path", "")
        title_text = os.path.basename(path) if path else "<unknown>"
        score = float(r.get("score", 0.0))
        chunk_idx = r.get("chunk_index", 0)
        snippet_full = r.get("text", "") or ""

        # Title (clickable)
        title = ttk.Label(frame, text=title_text, style="Title.TLabel", cursor="hand2")
        title.pack(anchor="w")
        if path:
            title.bind("<Button-1>", lambda e, p=path: system_open(p))

        # Path
        ttk.Label(frame, text=path, style="Path.TLabel").pack(anchor="w", pady=(2, 0))

        # Snippet with expand/collapse
        max_chars = 600
        is_expanded = {"v": False}
        text_var = tk.StringVar(value=self._truncate(snippet_full, max_chars))
        snippet = ttk.Label(frame, textvariable=text_var, wraplength=980, justify=tk.LEFT)
        snippet.pack(anchor="w", pady=(6, 2))

        def toggle():
            is_expanded["v"] = not is_expanded["v"]
            if is_expanded["v"]:
                text_var.set(snippet_full)
                btn.configure(text="Collapse")
            else:
                text_var.set(self._truncate(snippet_full, max_chars))
                btn.configure(text="Expand")

        btn = ttk.Button(frame, text="Expand", width=10, command=toggle)
        btn.pack(anchor="w")

        # Meta line
        ttk.Label(
            frame,
            text=f"chunk {chunk_idx} • score {(score*100):.1f}%",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(6, 0))

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1] + "…"

    # ---------- Queue/Status ----------
    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "log":
                    self._set_status(str(payload))
                elif kind == "results":
                    self._render_results(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
