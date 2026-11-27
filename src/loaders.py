"""Separate loaders for CSV, Markdown and HTML sources.

Each loader returns raw, un-chunked records: a list of dicts with a ``text``
field (human-readable text used for embedding) plus a ``metadata`` dict that
preserves the structured fields the capstone requires (product_id, category,
price, rating, seller_id, warranty, headings/sections, etc). Chunking is a
separate step performed by ``src.chunkers`` so loaders stay simple and testable
on their own.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
KB_DIR = ROOT / "knowledge_base"


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_product_catalogue(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load `product_catalogue.csv`. One row -> one record, metadata preserved."""
    path = path or (DATA_DIR / "product_catalogue.csv")
    records = []
    for row in _read_csv(path):
        text = (
            f"{row['name']} by {row['brand']} ({row['category']} > {row['subcategory']}). "
            f"Price: \u20b9{row['price_inr']}. Specs: {row['specs']}. "
            f"Rating: {row['rating']} from {row['review_count']} reviews. "
            f"Warranty: {row['warranty_months']} months. "
            f"Sold by {row['seller_id']} (verified: {row['seller_verified']}). "
            f"Tags: {row['tags']}."
        )
        records.append(
            {
                "text": text,
                "source_file": "product_catalogue.csv",
                "source_ref": row["product_id"],
                "doc_type": "product",
                "metadata": {
                    "product_id": row["product_id"],
                    "name": row["name"],
                    "brand": row["brand"],
                    "category": row["category"],
                    "subcategory": row["subcategory"],
                    "price_inr": float(row["price_inr"]),
                    "specs": row["specs"],
                    "rating": float(row["rating"]),
                    "review_count": int(row["review_count"]),
                    "seller_id": row["seller_id"],
                    "seller_verified": row["seller_verified"],
                    "warranty_months": int(row["warranty_months"]),
                    "tags": row["tags"],
                },
            }
        )
    return records


def load_review_summaries(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load `product_review_summaries.csv`. One row -> one record."""
    path = path or (DATA_DIR / "product_review_summaries.csv")
    records = []
    for row in _read_csv(path):
        text = (
            f"Review summary for {row['product_name']} ({row['category']}): "
            f"rating {row['rating']} across {row['review_count']} reviews. "
            f"Positive themes: {row['positive_themes']}. "
            f"Critical themes: {row['critical_themes']}. "
            f"{row['recommendation_reasoning']}"
        )
        records.append(
            {
                "text": text,
                "source_file": "product_review_summaries.csv",
                "source_ref": row["product_id"],
                "doc_type": "review",
                "metadata": {
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "rating": float(row["rating"]),
                    "review_count": int(row["review_count"]),
                    "positive_themes": row["positive_themes"],
                    "critical_themes": row["critical_themes"],
                },
            }
        )
    return records


# ---------------------------------------------------------------------------
# Markdown loader (heading-preserving)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def load_markdown(path: Path) -> List[Dict[str, Any]]:
    """Split a Markdown file into sections by heading, preserving heading text
    and level as metadata. A "section" runs from one heading to the next.
    """
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()

    sections: List[Dict[str, Any]] = []
    current_heading = path.stem.replace("_", " ").title()
    current_level = 0
    buffer: List[str] = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            sections.append({"heading": current_heading, "level": current_level, "body": body})

    for line in lines:
        match = _HEADING_RE.match(line.strip())
        if match:
            flush()
            buffer = []
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
        else:
            buffer.append(line)
    flush()

    records = []
    for i, sec in enumerate(sections):
        text = f"{sec['heading']}: {sec['body']}"
        records.append(
            {
                "text": text,
                "source_file": path.name,
                "source_ref": sec["heading"],
                "doc_type": "markdown",
                "metadata": {
                    "heading": sec["heading"],
                    "level": sec["level"],
                    "section_index": i,
                    "file": path.name,
                },
            }
        )
    return records


def load_all_markdown(kb_dir: Path | None = None) -> List[Dict[str, Any]]:
    kb_dir = kb_dir or KB_DIR
    records = []
    for path in sorted(kb_dir.glob("*.md")):
        records.extend(load_markdown(path))
    return records


# ---------------------------------------------------------------------------
# HTML loader (BeautifulSoup, FAQ-shaped: <section><h2>category</h2><h3>question</h3><p>answer</p></section>)
# ---------------------------------------------------------------------------

def load_html_faq(path: Path | None = None) -> List[Dict[str, Any]]:
    path = path or (KB_DIR / "buyer_faq.html")
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
    records = []
    for i, section in enumerate(soup.find_all("section")):
        h2 = section.find("h2")
        h3 = section.find("h3")
        p = section.find("p")
        category = h2.get_text(strip=True) if h2 else "General"
        question = h3.get_text(strip=True) if h3 else f"FAQ {i}"
        answer = p.get_text(strip=True) if p else ""
        text = f"{category} FAQ - {question}: {answer}"
        records.append(
            {
                "text": text,
                "source_file": path.name,
                "source_ref": question,
                "doc_type": "html",
                "metadata": {
                    "category": category,
                    "question": question,
                    "answer": answer,
                    "faq_index": i,
                    "file": path.name,
                },
            }
        )
    return records


def load_all_knowledge_base(kb_dir: Path | None = None) -> List[Dict[str, Any]]:
    """Ingest every file under knowledge_base/: Markdown via load_markdown, HTML via load_html_faq."""
    kb_dir = kb_dir or KB_DIR
    records = load_all_markdown(kb_dir)
    faq_path = kb_dir / "buyer_faq.html"
    if faq_path.exists():
        records.extend(load_html_faq(faq_path))
    return records
