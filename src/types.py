"""Shared dataclasses used across loaders, chunkers, retrieval and the chain."""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    """A single retrievable unit of text with metadata and a citation-ready source tag."""

    chunk_id: str
    text: str
    source_file: str          # e.g. "product_catalogue.csv" or "returns_and_refunds_policy.md"
    source_ref: str           # e.g. "SS0001" or "2. Eligibility" -- goes after the "|" in a citation
    doc_type: str             # "product" | "review" | "markdown" | "html"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        return f"[source: {self.source_file} | {self.source_ref}]"
