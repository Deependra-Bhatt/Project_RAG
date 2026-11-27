"""Persisted vector store.

Uses FAISS (`IndexFlatIP` over normalized vectors, i.e. cosine similarity) if
the `faiss` package is importable, otherwise a numpy brute-force cosine
fallback with an identical on-disk layout so callers never need to know which
backend is active. Chroma is not used here to avoid an extra server/process
dependency, but the on-disk format keeps both backends compatible with the
"persisted FAISS (or Chroma) vectors" requirement.

Index reuse: `VectorStore.load()` reads `vectors.npy` + `metadata.json` back
from disk without touching the embedding model, so restarting the app or CLI
does not re-embed the corpus.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.types import Chunk

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"


class VectorStore:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or (ARTIFACTS_DIR / "index")
        self.vectors: np.ndarray | None = None
        self.chunks: List[Chunk] = []
        self._faiss_index = None
        self.backend_name = "numpy"
        try:
            import faiss  # noqa: F401

            self._faiss_available = True
        except Exception:
            self._faiss_available = False

    # -- build --------------------------------------------------------
    def build(self, chunks: List[Chunk], vectors: np.ndarray) -> None:
        assert len(chunks) == vectors.shape[0], "chunks/vectors length mismatch"
        self.chunks = chunks
        self.vectors = vectors.astype("float32")
        if self._faiss_available:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss_index = index
            self.backend_name = "faiss"
        else:
            self.backend_name = "numpy"

    # -- persistence ----------------------------------------------------
    def save(self, tfidf_state: Any = None, backend_name: str = "") -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        np.save(self.index_dir / "vectors.npy", self.vectors)
        metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source_file": c.source_file,
                "source_ref": c.source_ref,
                "doc_type": c.doc_type,
                "metadata": c.metadata,
            }
            for c in self.chunks
        ]
        with (self.index_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f)
        with (self.index_dir / "embedding_backend.json").open("w", encoding="utf-8") as f:
            json.dump({"backend_name": backend_name}, f)
        if tfidf_state is not None:
            with (self.index_dir / "tfidf_state.pkl").open("wb") as f:
                pickle.dump(tfidf_state, f)

    def load(self) -> Tuple[bool, str]:
        """Load a previously persisted index. Returns (loaded, embedding_backend_name)."""
        vec_path = self.index_dir / "vectors.npy"
        meta_path = self.index_dir / "metadata.json"
        if not (vec_path.exists() and meta_path.exists()):
            return False, ""
        self.vectors = np.load(vec_path)
        with meta_path.open(encoding="utf-8") as f:
            raw = json.load(f)
        self.chunks = [
            Chunk(
                chunk_id=r["chunk_id"],
                text=r["text"],
                source_file=r["source_file"],
                source_ref=r["source_ref"],
                doc_type=r["doc_type"],
                metadata=r["metadata"],
            )
            for r in raw
        ]
        if self._faiss_available:
            import faiss

            index = faiss.IndexFlatIP(self.vectors.shape[1])
            index.add(self.vectors)
            self._faiss_index = index
            self.backend_name = "faiss"
        else:
            self.backend_name = "numpy"

        backend_name = ""
        bpath = self.index_dir / "embedding_backend.json"
        if bpath.exists():
            with bpath.open(encoding="utf-8") as f:
                backend_name = json.load(f).get("backend_name", "")
        return True, backend_name

    def load_tfidf_state(self):
        path = self.index_dir / "tfidf_state.pkl"
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        return None

    # -- search -----------------------------------------------------------
    def search(self, query_vector: np.ndarray, k: int, allowed_indices: List[int] | None = None) -> List[Tuple[int, float]]:
        """Return list of (index_into_self.chunks, score) sorted descending."""
        if self.vectors is None or len(self.chunks) == 0:
            return []
        q = query_vector.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        if allowed_indices is not None:
            if not allowed_indices:
                return []
            sub_vectors = self.vectors[allowed_indices]
            scores = sub_vectors @ q[0]
            order = np.argsort(-scores)[:k]
            return [(allowed_indices[i], float(scores[i])) for i in order]

        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(q, min(k, len(self.chunks)))
            return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i != -1]

        scores = self.vectors @ q[0]
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]
