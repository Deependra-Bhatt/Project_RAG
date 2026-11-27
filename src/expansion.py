"""Expand every incoming query into up to two deduplicated variants.

The synonym map targets the e-commerce vocabulary mismatches called out in the
build prompt: earphones/earbuds/TWS, refund/reimbursement, return/send back,
smartwatch/wearable. Expansion is intentionally shallow (rule-based, no model
call) so it works with zero dependencies and is fast enough to run on every
query.
"""
from __future__ import annotations

import re
from typing import List

SYNONYM_GROUPS: List[List[str]] = [
    ["earphones", "earbuds", "earbud", "tws", "wireless earbuds"],
    ["refund", "reimbursement", "money back"],
    ["return", "send back", "returning"],
    ["smartwatch", "wearable", "smart watch"],
    ["cancel", "cancellation"],
    ["fake", "counterfeit", "duplicate"],
    ["warranty", "guarantee"],
    ["delivery", "shipping"],
    ["price", "cost"],
]

_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


def expand_query(query: str) -> List[str]:
    """Return a deduplicated list of at most two query variants: the original
    query and one synonym-substituted variant (if any synonym applies).
    """
    original = query.strip()
    if not original:
        return [original]

    lower = original.lower()
    variant = None
    for group in SYNONYM_GROUPS:
        for term in group:
            if term in lower:
                # swap in the first *other* term from the group
                replacement = next((t for t in group if t != term), None)
                if replacement:
                    variant = re.sub(re.escape(term), replacement, lower, count=1)
                break
        if variant:
            break

    variants = [original]
    if variant and variant.strip().lower() != original.strip().lower():
        variants.append(variant)

    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for v in variants:
        key = v.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(v)
    return deduped[:2]
