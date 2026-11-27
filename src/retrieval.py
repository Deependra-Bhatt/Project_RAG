"""Retrieval layer: applies metadata filters, expands the query into
variants, searches the vector store, and (for comparative/recommendation
questions) widens the candidate pool to >= 8 chunks before applying
deterministic price/rating/spec checks. Every returned chunk carries a
ready-made citation string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.embeddings import EmbeddingBackend
from src.expansion import expand_query
from src.types import Chunk
from src.vectorstore import VectorStore

COMPARATIVE_KEYWORDS = (
    "best", "compare", "comparison", "vs", "versus", "recommend", "recommendation",
    "which one", "top", "cheapest", "options", "alternatives",
)

MIN_COMPARATIVE_K = 8
DEFAULT_K = 8


@dataclass
class Filters:
    category: Optional[str] = None
    max_price: Optional[float] = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float

    @property
    def citation(self) -> str:
        return self.chunk.citation()


def is_comparative(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in COMPARATIVE_KEYWORDS)


class Retriever:
    def __init__(self, vectorstore: VectorStore, embedder: EmbeddingBackend):
        self.vs = vectorstore
        self.embedder = embedder

    def _filtered_indices(self, filters: Filters) -> Optional[List[int]]:
        """Pre-filter product/review chunks by category and price ceiling.
        Non-product chunks (policy/FAQ docs) are always kept in scope since a
        category/price filter should not hide policy or FAQ evidence.
        Returns None if no filter is active (search the whole store).
        """
        if not filters.category and filters.max_price is None:
            return None

        allowed = []
        for i, chunk in enumerate(self.vs.chunks):
            if chunk.doc_type not in ("product", "product_group", "review"):
                allowed.append(i)
                continue
            meta = chunk.metadata
            if filters.category and meta.get("category") != filters.category:
                continue
            if filters.max_price is not None:
                price = meta.get("price_inr")
                if price is not None and price > filters.max_price:
                    continue
            allowed.append(i)
        return allowed

    def retrieve(
        self,
        query: str,
        filters: Optional[Filters] = None,
        k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        filters = filters or Filters()
        comparative = is_comparative(query)
        k = k or (MIN_COMPARATIVE_K if comparative else DEFAULT_K)
        if comparative:
            k = max(k, MIN_COMPARATIVE_K)

        variants = expand_query(query)
        allowed = self._filtered_indices(filters)

        best_by_index: Dict[int, float] = {}
        for variant in variants:
            qvec = self.embedder.encode([variant])[0]
            hits = self.vs.search(qvec, k=k, allowed_indices=allowed)
            for idx, score in hits:
                if idx not in best_by_index or score > best_by_index[idx]:
                    best_by_index[idx] = score

        ranked = sorted(best_by_index.items(), key=lambda kv: -kv[1])[:k]
        results = [RetrievedChunk(chunk=self.vs.chunks[idx], score=score) for idx, score in ranked]

        if comparative:
            results = self._deterministic_rank(results, filters)
        return results

    def _deterministic_rank(self, results: List[RetrievedChunk], filters: Filters) -> List[RetrievedChunk]:
        """Comparative/recommendation questions must not rely on embedding
        similarity alone: re-rank product-type candidates deterministically
        by (price ceiling compliance, rating desc, price asc).
        """
        def sort_key(rc: RetrievedChunk):
            meta = rc.chunk.metadata
            if rc.chunk.doc_type not in ("product", "product_group", "review"):
                return (1, 0, 0)  # keep non-product evidence but after ranked products
            price = meta.get("price_inr", float("inf"))
            rating = meta.get("rating", 0)
            over_budget = 1 if (filters.max_price is not None and price and price > filters.max_price) else 0
            return (0, over_budget, -rating, price if isinstance(price, (int, float)) else 0)

        return sorted(results, key=sort_key)
