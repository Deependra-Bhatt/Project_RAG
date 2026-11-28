"""Evaluation harness for CatalogueIQ.

This produces the baseline vs. post-improvement numbers referenced in
`evaluation_report.md`. It intentionally does NOT depend on the `ragas`
package or an LLM judge (OpenAI/Anthropic), because this build environment
has no network access and the app must also be gradeable with no secrets.
Instead it implements small, transparent proxy metrics for the same three
RAGAS dimensions:

- faithfulness_proxy: did the answer carry a citation, and does every citation
  actually point at a chunk that was retrieved (no invented sources)?
- answer_relevancy_proxy: lexical overlap between the query's content words
  and the answer's content words (Jaccard over non-stopword tokens).
- context_precision_proxy: fraction of retrieved chunks that share a content
  word with the query (signal vs. noise in what was retrieved).

Two configurations are compared:
- baseline: Strategy B chunks (grouped products + fixed-size text windows),
  no query expansion, fixed k=5, no comparative widening/re-ranking, no
  "no evidence" lexical gate.
- improved: Strategy A chunks (row + recursive section chunks) with query
  expansion, comparative widening to >=8, deterministic re-ranking, and the
  lexical no-evidence gate -- i.e. exactly what `app.py` uses today.

Run with:  python -m evaluation.run_eval
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chain import RagChain  # noqa: E402
from src.chunkers import build_chunks  # noqa: E402
from src.embeddings import EmbeddingBackend  # noqa: E402
from src.pipeline import get_retriever, load_records  # noqa: E402
from src.retrieval import Filters, Retriever, RetrievedChunk, is_comparative  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

_STOPWORDS = {
    "the", "a", "an", "is", "are", "do", "you", "can", "i", "of", "for", "to", "in",
    "on", "and", "or", "what", "how", "does", "did", "it", "my", "me", "with",
    "this", "that", "your", "have", "has", "will", "be", "was", "were",
}
_WORD_RE = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


class BaselineRetriever:
    """A deliberately weaker retriever used only to measure improvement:
    Strategy B chunks, single query variant (no expansion), fixed k=5, no
    metadata pre-filtering, no comparative widening, no re-ranking.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        products, reviews, kb = load_records()
        chunks = build_chunks("b", products, reviews, kb)
        self.embedder = EmbeddingBackend()
        self.embedder.fit([c.text for c in chunks])
        vectors = self.embedder.encode([c.text for c in chunks])
        self.vs = VectorStore(index_dir=index_dir)
        self.vs.build(chunks, vectors)

    def retrieve(self, query: str, filters: Filters | None = None, k: int = 5) -> List[RetrievedChunk]:
        qvec = self.embedder.encode([query])[0]
        hits = self.vs.search(qvec, k=5)
        return [RetrievedChunk(chunk=self.vs.chunks[i], score=s) for i, s in hits]


class BaselineChain:
    """Mirrors RagChain.answer() but skips expansion/widening/rerank/no-evidence gate."""

    def __init__(self, retriever: BaselineRetriever):
        self.retriever = retriever

    def answer(self, query: str, persona: str, filters: Filters | None = None):
        results = self.retriever.retrieve(query, filters=filters)
        citations = [rc.citation for rc in results]
        if not results:
            answer = "I couldn't find this in the ShopSmart catalogue or knowledge base."
            mode = "no_evidence"
        else:
            lines = [f"- {rc.chunk.text[:200]} {rc.citation}" for rc in results[:5]]
            answer = "Here's what I found:\n" + "\n".join(lines)
            mode = "retrieval_only"

        class R:
            pass

        r = R()
        r.answer = answer
        r.citations = citations
        r.retrieved = results
        r.mode = mode
        return r


@dataclass
class Metrics:
    faithfulness: float
    answer_relevancy: float
    context_precision: float


def faithfulness_proxy(query: str, result) -> float:
    if result.mode == "no_evidence":
        # correctly abstaining is faithful as long as nothing was invented
        return 1.0 if not result.citations else 0.0
    if not result.citations:
        return 0.0
    retrieved_refs = {rc.chunk.source_ref for rc in result.retrieved}
    matched = sum(1 for c in result.citations if any(ref in c for ref in retrieved_refs))
    return matched / len(result.citations)


def answer_relevancy_proxy(query: str, result) -> float:
    q_words = _content_words(query)
    a_words = _content_words(result.answer)
    if not q_words:
        return 1.0
    overlap = q_words & a_words
    return len(overlap) / len(q_words)


def context_precision_proxy(query: str, result) -> float:
    if not result.retrieved:
        return 0.0
    q_words = _content_words(query)
    if not q_words:
        return 1.0
    relevant = sum(1 for rc in result.retrieved if q_words & _content_words(rc.chunk.text))
    return relevant / len(result.retrieved)


def run(questions: List[Dict[str, Any]], improved_chain, baseline_chain) -> Dict[str, Dict[str, float]]:
    improved_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}
    baseline_scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}
    per_question = []

    for q in questions:
        filters = Filters(**q.get("filters", {})) if q.get("filters") else None

        improved_result = improved_chain.answer(q["query"], q["persona"], filters=filters)
        baseline_result = baseline_chain.answer(q["query"], q["persona"], filters=filters)

        i_f, i_a, i_c = (
            faithfulness_proxy(q["query"], improved_result),
            answer_relevancy_proxy(q["query"], improved_result),
            context_precision_proxy(q["query"], improved_result),
        )
        b_f, b_a, b_c = (
            faithfulness_proxy(q["query"], baseline_result),
            answer_relevancy_proxy(q["query"], baseline_result),
            context_precision_proxy(q["query"], baseline_result),
        )

        improved_scores["faithfulness"].append(i_f)
        improved_scores["answer_relevancy"].append(i_a)
        improved_scores["context_precision"].append(i_c)
        baseline_scores["faithfulness"].append(b_f)
        baseline_scores["answer_relevancy"].append(b_a)
        baseline_scores["context_precision"].append(b_c)

        per_question.append(
            {
                "id": q["id"], "type": q["type"], "query": q["query"],
                "baseline": {"faithfulness": b_f, "answer_relevancy": b_a, "context_precision": b_c},
                "improved": {"faithfulness": i_f, "answer_relevancy": i_a, "context_precision": i_c},
            }
        )

    def avg(d):
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in d.items()}

    return {"baseline": avg(baseline_scores), "improved": avg(improved_scores), "per_question": per_question}


def main():
    questions = json.loads((ROOT / "evaluation" / "eval_questions.json").read_text(encoding="utf-8"))

    improved_retriever = get_retriever(strategy="a")
    improved_chain = RagChain(improved_retriever)

    baseline_retriever = BaselineRetriever(index_dir=ROOT / "artifacts" / "baseline_index")
    baseline_chain = BaselineChain(baseline_retriever)

    results = run(questions, improved_chain, baseline_chain)

    print(json.dumps({"baseline": results["baseline"], "improved": results["improved"]}, indent=2))

    out_path = ROOT / "evaluation" / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote detailed per-question results to {out_path}")


if __name__ == "__main__":
    main()
