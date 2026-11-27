"""CLI inspection command.

    python -m src.cli build-index [--strategy a|b]
    python -m src.cli inspect "wireless earbuds under 2000" [--category Electronics] [--max-price 2000] [--k 5]
    python -m src.cli compare-strategies "wireless earbuds under 2000"
"""
from __future__ import annotations

import argparse

from src.chunkers import build_chunks
from src.pipeline import build_index, get_retriever, load_records
from src.retrieval import Filters


def cmd_build_index(args):
    vs, embedder = build_index(strategy=args.strategy)
    print(f"Built index: {len(vs.chunks)} chunks, backend={embedder.backend_name}, store={vs.backend_name}")


def cmd_inspect(args):
    retriever = get_retriever(force_rebuild=args.rebuild)
    filters = Filters(category=args.category, max_price=args.max_price)
    results = retriever.retrieve(args.query, filters=filters, k=args.k)
    print(f"Query: {args.query!r}  |  filters: category={args.category} max_price={args.max_price}")
    print(f"Retrieved {len(results)} chunks:\n")
    for rc in results:
        print(f"  score={rc.score:.3f}  {rc.citation}")
        print(f"    {rc.chunk.text[:200]}")
        print()


def cmd_compare_strategies(args):
    products, reviews, kb_records = load_records()
    for strategy in ("a", "b"):
        chunks = build_chunks(strategy, products, reviews, kb_records)
        lengths = [len(c.text) for c in chunks]
        print(f"Strategy {strategy}: {len(chunks)} chunks, avg length {sum(lengths)/len(lengths):.0f} chars, "
              f"min {min(lengths)}, max {max(lengths)}")


def main():
    parser = argparse.ArgumentParser(description="CatalogueIQ inspection CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index")
    p_build.add_argument("--strategy", default="a", choices=["a", "b"])
    p_build.set_defaults(func=cmd_build_index)

    p_inspect = sub.add_parser("inspect")
    p_inspect.add_argument("query")
    p_inspect.add_argument("--category", default=None)
    p_inspect.add_argument("--max-price", dest="max_price", type=float, default=None)
    p_inspect.add_argument("--k", type=int, default=5)
    p_inspect.add_argument("--rebuild", action="store_true")
    p_inspect.set_defaults(func=cmd_inspect)

    p_compare = sub.add_parser("compare-strategies")
    p_compare.add_argument("query", nargs="?", default="wireless earbuds under 2000")
    p_compare.set_defaults(func=cmd_compare_strategies)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
