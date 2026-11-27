"""End-to-end ingestion pipeline used by both the CLI and the Streamlit app.

`get_retriever()` is the single entry point: it loads a persisted index if
one exists, otherwise builds one from `data/` and `knowledge_base/` and
persists it, then returns a ready-to-use `Retriever`.

Embeddings and vector stores are cached efficiently to avoid expensive
reinitialization on repeated calls.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from src.chunkers import build_chunks
from src.embeddings import EmbeddingBackend
from src.loaders import load_all_knowledge_base, load_product_catalogue, load_review_summaries
from src.retrieval import Retriever
from src.vectorstore import VectorStore

ROOT = Path(__file__).resolve().parents[1]

# Module-level cache for expensive resources to avoid reinitialization
_cached_retriever: Retriever | None = None
_cache_index_dir: Path | None = None


def load_records():
    products = load_product_catalogue()
    reviews = load_review_summaries()
    kb_records = load_all_knowledge_base()
    return products, reviews, kb_records


def build_index(strategy: str = "a", index_dir: Path | None = None) -> Tuple[VectorStore, EmbeddingBackend]:
    products, reviews, kb_records = load_records()
    chunks = build_chunks(strategy, products, reviews, kb_records)

    embedder = EmbeddingBackend()
    embedder.fit([c.text for c in chunks])
    vectors = embedder.encode([c.text for c in chunks])

    vs = VectorStore(index_dir=index_dir)
    vs.build(chunks, vectors)
    tfidf_state = embedder._tfidf if embedder._st_model is None else None
    vs.save(tfidf_state=tfidf_state, backend_name=embedder.backend_name)
    return vs, embedder


def get_retriever(strategy: str = "a", index_dir: Path | None = None, force_rebuild: bool = False) -> Retriever:
    """Get or create a Retriever instance. Caches at the module level to avoid
    expensive reinitialization of embeddings and vector stores on repeated calls.
    
    The cache is invalidated only if the index_dir changes or force_rebuild is True.
    """
    global _cached_retriever, _cache_index_dir
    
    # Return cached retriever if parameters haven't changed
    if (
        _cached_retriever is not None
        and not force_rebuild
        and _cache_index_dir == index_dir
    ):
        return _cached_retriever
    
    vs = VectorStore(index_dir=index_dir)
    embedder = EmbeddingBackend()

    loaded, backend_name = (False, "") if force_rebuild else vs.load()
    if loaded:
        if embedder._st_model is None:
            tfidf_state = vs.load_tfidf_state()
            if tfidf_state is not None:
                embedder._tfidf = tfidf_state
        _cached_retriever = Retriever(vs, embedder)
        _cache_index_dir = index_dir
        return _cached_retriever

    vs, embedder = build_index(strategy=strategy, index_dir=index_dir)
    _cached_retriever = Retriever(vs, embedder)
    _cache_index_dir = index_dir
    return _cached_retriever
