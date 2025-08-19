from __future__ import annotations

"""Embeddings utilities for GGUF models via llama-cpp-python.

This module provides a small class to load/unload a GGUF embedding model and
generate embeddings from strings. It relies on `llama_cpp` (llama-cpp-python).

Comments are in English as requested.
"""

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence
import gc
import os


try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional dependency at runtime
    Llama = None  # type: ignore


@dataclass
class ModelConfig:
    """Configuration for the GGUF model runtime."""

    model_path: str
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    n_batch: int = 512
    # Embedding flag must be True to enable embedding API in llama.cpp
    embedding: bool = True
    # Reduce llama.cpp logging verbosity (set to True for debugging)
    verbose: bool = False


class GGUFEmbedder:
    """Thin wrapper over llama-cpp to produce embeddings.

    Usage:
        emb = GGUFEmbedder()
        emb.load(ModelConfig(model_path="/path/to/model.gguf"))
        vec = emb.embed_text("hello world")
        emb.unload()
    """

    def __init__(self) -> None:
        self._llama: Optional["Llama"] = None
        self._cfg: Optional[ModelConfig] = None
        self._last_dim: Optional[int] = None

    @property
    def is_loaded(self) -> bool:
        return self._llama is not None

    @property
    def dim(self) -> Optional[int]:
        """Return last observed embedding dimension (if any)."""

        return self._last_dim

    def load(self, cfg: ModelConfig) -> None:
        """Load a GGUF model for embeddings."""

        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Please `pip install llama-cpp-python`."
            )
        if not os.path.exists(cfg.model_path):
            raise FileNotFoundError(cfg.model_path)
        # Clean previous instance if any
        if self._llama is not None:
            self.unload()
        # Reduce logging noise unless explicitly overridden
        if not cfg.verbose and os.getenv("LLAMA_LOG_LEVEL") is None:
            os.environ["LLAMA_LOG_LEVEL"] = "error"
        # Instantiate Llama with embedding enabled
        self._llama = Llama(
            model_path=cfg.model_path,
            n_ctx=cfg.n_ctx,
            n_threads=cfg.n_threads or 0,
            n_batch=cfg.n_batch,
            embedding=cfg.embedding,
            verbose=cfg.verbose,
        )
        self._cfg = cfg

    def unload(self) -> None:
        """Unload the model and free memory."""

        self._cfg = None
        self._llama = None
        # Encourage freeing native memory ASAP
        gc.collect()

    def embed_text(self, text: str) -> List[float]:
        """Return embedding vector for a single string."""

        if self._llama is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        # llama.cpp API accepts string or list of strings. We use a single input.
        out = self._llama.create_embedding(text)  # type: ignore[attr-defined]
        vec = out["data"][0]["embedding"]
        # Track dimension for convenience.
        self._last_dim = len(vec)
        return vec

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Return embedding vectors for multiple strings (batched)."""

        if self._llama is None:
            raise RuntimeError("Model is not loaded. Call load() first.")
        if not texts:
            return []
        out = self._llama.create_embedding(texts)  # type: ignore[attr-defined]
        vecs = [item["embedding"] for item in out["data"]]
        self._last_dim = len(vecs[0]) if vecs else None
        return vecs
