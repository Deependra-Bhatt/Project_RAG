"""Two chunking strategies over the loaded records.

Strategy A ("row_and_recursive"): one product/review CSV row stays exactly one
chunk (preserves product_id / category / price / rating / seller / warranty
as metadata for precise citation and filtering); long knowledge-base sections
are recursively split on paragraph -> sentence boundaries with overlap so no
chunk exceeds a target size.

Strategy B ("grouped_and_fixed"): multiple product rows are grouped (by
category+subcategory) into a single denser chunk, and knowledge-base text is
split into fixed-size character windows regardless of heading/paragraph
boundaries.

`inspect_strategies()` runs a handful of sample queries against both and is
used by the CLI (`python -m src.cli compare-strategies`) and by
`evaluation_report.md` to justify picking Strategy A.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.types import Chunk

TARGET_SIZE = 480   # characters
OVERLAP = 80


def _recursive_split(text: str, target: int = TARGET_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split on paragraph boundaries first, then sentences, then hard-wrap, with overlap."""
    if len(text) <= target:
        return [text] if text.strip() else []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [s.strip() for s in text.replace(". ", ".\n").split("\n") if s.strip()]

    chunks: List[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > target:
            chunks.append(buf)
            # carry the tail of the previous chunk forward as overlap
            buf = buf[-overlap:] + " " + para if overlap else para
        else:
            buf = (buf + " " + para).strip() if buf else para
    if buf:
        chunks.append(buf)

    # hard-wrap anything still too long (e.g. one giant paragraph)
    final: List[str] = []
    for c in chunks:
        while len(c) > target * 1.5:
            final.append(c[: target])
            c = c[target - overlap :]
        final.append(c)
    return [c for c in final if c.strip()]


def strategy_a_row_and_recursive(
    products: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    kb_records: List[Dict[str, Any]],
) -> List[Chunk]:
    chunks: List[Chunk] = []

    for rec in products:
        chunks.append(
            Chunk(
                chunk_id=f"product:{rec['source_ref']}",
                text=rec["text"],
                source_file=rec["source_file"],
                source_ref=rec["source_ref"],
                doc_type=rec["doc_type"],
                metadata=rec["metadata"],
            )
        )

    for rec in reviews:
        chunks.append(
            Chunk(
                chunk_id=f"review:{rec['source_ref']}",
                text=rec["text"],
                source_file=rec["source_file"],
                source_ref=rec["source_ref"],
                doc_type=rec["doc_type"],
                metadata=rec["metadata"],
            )
        )

    for rec in kb_records:
        if rec["doc_type"] == "html":
            # FAQ entries are already short/atomic -- keep as one chunk each.
            chunks.append(
                Chunk(
                    chunk_id=f"html:{rec['source_file']}:{rec['metadata']['faq_index']}",
                    text=rec["text"],
                    source_file=rec["source_file"],
                    source_ref=rec["source_ref"],
                    doc_type=rec["doc_type"],
                    metadata=rec["metadata"],
                )
            )
            continue
        pieces = _recursive_split(rec["text"])
        for j, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=f"md:{rec['source_file']}:{rec['metadata']['section_index']}:{j}",
                    text=piece,
                    source_file=rec["source_file"],
                    source_ref=rec["source_ref"],
                    doc_type=rec["doc_type"],
                    metadata={**rec["metadata"], "split_index": j},
                )
            )
    return chunks


def strategy_b_grouped_and_fixed(
    products: List[Dict[str, Any]],
    reviews: List[Dict[str, Any]],
    kb_records: List[Dict[str, Any]],
    group_size: int = 5,
    window: int = 400,
) -> List[Chunk]:
    chunks: List[Chunk] = []

    # Group products by (category, subcategory), then bundle `group_size` rows per chunk.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in products:
        key = f"{rec['metadata']['category']}|{rec['metadata']['subcategory']}"
        groups.setdefault(key, []).append(rec)

    for key, recs in groups.items():
        for i in range(0, len(recs), group_size):
            batch = recs[i : i + group_size]
            text = " || ".join(r["text"] for r in batch)
            ids = ",".join(r["source_ref"] for r in batch)
            chunks.append(
                Chunk(
                    chunk_id=f"group:{key}:{i}",
                    text=text,
                    source_file="product_catalogue.csv",
                    source_ref=ids,
                    doc_type="product_group",
                    metadata={"category": batch[0]["metadata"]["category"], "product_ids": [r["source_ref"] for r in batch]},
                )
            )

    for rec in reviews:
        chunks.append(
            Chunk(
                chunk_id=f"review:{rec['source_ref']}",
                text=rec["text"],
                source_file=rec["source_file"],
                source_ref=rec["source_ref"],
                doc_type=rec["doc_type"],
                metadata=rec["metadata"],
            )
        )

    # Fixed-size windows over the raw KB text, ignoring heading boundaries.
    for rec in kb_records:
        text = rec["text"]
        for j in range(0, len(text), window):
            piece = text[j : j + window]
            if not piece.strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"fixed:{rec['source_file']}:{j}",
                    text=piece,
                    source_file=rec["source_file"],
                    source_ref=rec.get("source_ref", rec["source_file"]),
                    doc_type=rec["doc_type"],
                    metadata=rec["metadata"],
                )
            )
    return chunks


def build_chunks(strategy: str, products, reviews, kb_records) -> List[Chunk]:
    if strategy == "a":
        return strategy_a_row_and_recursive(products, reviews, kb_records)
    if strategy == "b":
        return strategy_b_grouped_and_fixed(products, reviews, kb_records)
    raise ValueError(f"Unknown strategy: {strategy}")
