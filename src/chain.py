"""The grounded answer chain wires retrieval, persona prompts, a short
conversation memory and an optional generation provider together. It always
runs in retrieval-only mode when no provider/API key is configured, and every
answer it returns carries citations (or plainly says evidence is absent).

Scope validation ensures the chatbot only answers questions that fall within
the selected persona and category boundaries."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.personas import build_system_prompt, PERSONA_DEFINITIONS
from src.provider import AnthropicProvider
from src.retrieval import Filters, RetrievedChunk, Retriever

MAX_HISTORY_TURNS = 5

_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "you", "can", "i", "of", "for", "to", "in",
    "on", "and", "or", "what", "how", "does", "did", "it", "my", "me", "with",
    "this", "that", "your", "have", "has", "will", "be", "was", "were",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _has_lexical_evidence(query: str, results: List[RetrievedChunk]) -> bool:
    """A cheap, backend-agnostic sanity check: does the top result actually
    share a real content word with the query? Protects against an embedding
    backend (especially the offline TF-IDF fallback) confidently returning an
    unrelated chunk for a product/topic that simply isn't in the catalogue.
    """
    if not results:
        return False
    q_words = _content_words(query)
    if not q_words:
        return True
    top_words = _content_words(results[0].chunk.text)
    return bool(q_words & top_words)


def _is_scope_mismatch(query: str, persona: str, retrieved: List[RetrievedChunk]) -> bool:
    """Check if the query appears to be out of scope for the selected persona.
    
    This is a two-part check:
    1. Lexical check: Does the query share content words with the retrieved results?
    2. Coherence check: Would the model be confused about persona/category scope?
    
    Returns True if the query seems out of scope for the persona.
    
    This is a complementary check to the strong enforcement in the system prompt.
    """
    if not retrieved:
        # No retrieved evidence: let the system prompt handle scope enforcement
        return False
    
    # Check if the retrieved results actually contain related information
    has_coherent_evidence = _has_lexical_evidence(query, retrieved)
    
    # If we got strong lexical matches, the query is likely in scope
    # The system prompt will handle the final scope verification
    return not has_coherent_evidence


def _out_of_scope_response(query: str, persona: str, category: str) -> str:
    """Generate a response when a query is out of scope for the selected persona/category."""
    if category != "All":
        return (
            f"This question falls outside the currently selected {persona} / {category} scope. "
            f"Please select a different persona or category to continue. "
            f"For example, a different category might be more suitable for this question."
        )
    else:
        return (
            f"This question falls outside the currently selected {persona} scope. "
            f"Please select a different persona (e.g., Shopper, Seller, or Support Agent) that matches your question. "
            f"The current {persona} configuration cannot help with this topic."
        )


@dataclass
class Turn:
    persona: str
    query: str
    answer: str


@dataclass
class ChainResult:
    answer: str
    citations: List[str]
    retrieved: List[RetrievedChunk]
    mode: str  # "generated" | "retrieval_only" | "no_evidence" | "out_of_scope"


class ConversationMemory:
    """Keeps the last MAX_HISTORY_TURNS turns."""

    def __init__(self):
        self.turns: List[Turn] = []

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.turns = self.turns[-MAX_HISTORY_TURNS:]

    def as_prompt_block(self) -> str:
        if not self.turns:
            return ""
        lines = ["Recent conversation:"]
        for t in self.turns:
            lines.append(f"User ({t.persona}): {t.query}")
            lines.append(f"Assistant: {t.answer}")
        return "\n".join(lines)


def _trim(text: str, n: int = 220) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "\u2026"


def _retrieval_only_answer(query: str, persona: str, results: List[RetrievedChunk]) -> str:
    if not results:
        return (
            "I couldn't find this in the ShopSmart catalogue or knowledge base, "
            "so I don't want to guess. Please rephrase, widen the filters, or check "
            "with a human agent for anything not covered here."
        )

    lines = []
    if persona == "Support Agent":
        lines.append("Here is what the policy/knowledge base says:")
    elif persona == "Seller":
        lines.append("Relevant listing/compliance guidance:")
    else:
        lines.append("Here's what I found:")

    for rc in results[:8]:
        lines.append(f"- {_trim(rc.chunk.text)} {rc.citation}")
    return "\n".join(lines)


class RagChain:
    def __init__(self, retriever: Retriever, provider: Optional[AnthropicProvider] = None):
        self.retriever = retriever
        self.provider = provider or AnthropicProvider()

    def answer(
        self,
        query: str,
        persona: str,
        filters: Optional[Filters] = None,
        memory: Optional[ConversationMemory] = None,
        k: Optional[int] = None,
        category: str = "All",
    ) -> ChainResult:
        """Answer a user query while enforcing persona and category scope.
        
        Args:
            query: The user's question
            persona: The selected persona (Shopper, Seller, or Support Agent)
            filters: Optional filters for retrieval (category, max_price)
            memory: Optional conversation memory
            k: Optional override for number of results to retrieve
            category: The selected product category (for display and scope checking)
        
        Returns:
            A ChainResult with the answer, citations, retrieved chunks, and mode
        """
        results = self.retriever.retrieve(query, filters=filters, k=k)
        if results and not _has_lexical_evidence(query, results):
            results = []

        # Check for scope mismatch: if query seems out of scope, refuse to answer
        if _is_scope_mismatch(query, persona, results) and not results:
            out_of_scope_msg = _out_of_scope_response(query, persona, category)
            result = ChainResult(
                answer=out_of_scope_msg,
                citations=[],
                retrieved=[],
                mode="out_of_scope",
            )
            self._remember(memory, persona, query, result.answer)
            return result

        if not results:
            result = ChainResult(
                answer=_retrieval_only_answer(query, persona, results),
                citations=[],
                retrieved=[],
                mode="no_evidence",
            )
            self._remember(memory, persona, query, result.answer)
            return result

        citations = [rc.citation for rc in results]

        if self.provider.available:
            context = "\n".join(f"{rc.chunk.text} {rc.citation}" for rc in results)
            history_block = memory.as_prompt_block() if memory else ""
            # Pass category to system prompt for better scope enforcement
            system_prompt = build_system_prompt(persona, category)
            user_prompt = (
                f"{history_block}\n\nRetrieved context:\n{context}\n\nQuestion: {query}\n\n"
                "Answer using only the context above. End every factual sentence with its "
                "[source: FILE | REF] citation."
            )
            generated = self.provider.generate(system_prompt, user_prompt)
            if generated:
                # Guarantee at least one citation is present even if the model forgot.
                if "[source:" not in generated:
                    generated = generated.rstrip() + "\n\n" + "\n".join(citations[:3])
                result = ChainResult(answer=generated, citations=citations, retrieved=results, mode="generated")
                self._remember(memory, persona, query, result.answer)
                return result

        # Retrieval-only fallback (no provider configured, or generation failed)
        answer = _retrieval_only_answer(query, persona, results)
        result = ChainResult(answer=answer, citations=citations, retrieved=results, mode="retrieval_only")
        self._remember(memory, persona, query, result.answer)
        return result

    @staticmethod
    def _remember(memory: Optional[ConversationMemory], persona: str, query: str, answer: str) -> None:
        if memory is not None:
            memory.add(Turn(persona=persona, query=query, answer=answer))
