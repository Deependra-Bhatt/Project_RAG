"""Optional generation provider adapter.

The app must run with no secret in retrieval-only mode. This module only
activates real generation when an API key is configured *and* the SDK is
importable; otherwise `generate()` returns None and callers fall back to a
templated, citation-only answer built directly from retrieved chunks.
"""
from __future__ import annotations

import os
from dotenv import load_dotenv
from typing import List, Optional

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

class AnthropicProvider:
    model = "claude-sonnet-4-6"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self._client = None
        if self.api_key:
            try:
                import anthropic  # type: ignore

                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> Optional[str]:
        if not self.available:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            parts: List[str] = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
            return "\n".join(parts).strip() or None
        except Exception:
            return None
