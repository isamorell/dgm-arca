"""Phase 3 evaluation: Recall@k and MRR of each retriever on your gold set.

The gold set is a JSONL file (``rag/gold/example_gold.jsonl`` shows the format) with a
question and the id(s) of the chunk(s) that answer it. Fifty questions is the minimum;
write them yourselves while reading the corpus, and note the ones where two chunks are
equally valid: they will tell you a lot about your chunking.

Run::

    uv run python -m rag.evaluate --gold rag/gold/gold.jsonl --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / position of the first relevant chunk (0 when none is retrieved)."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def evaluate_retriever(retriever, gold: list[dict], k: int) -> dict[str, float]:
    recalls, rrs = [], []
    for item in gold:
        ranked = [hit.chunk.id for hit in retriever.search(item["question"], top_k=max(k, 10))]
        relevant = set(item["relevant_chunk_ids"])
        recalls.append(recall_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
    n = max(len(gold), 1)
    return {f"recall@{k}": sum(recalls) / n, "mrr": sum(rrs) / n, "n_questions": len(gold)}


def load_gold(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/example_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="reports/phase3_eval.json")
    args = parser.parse_args()
    gold = load_gold(args.gold)
    print(f"{len(gold)} gold questions loaded; results will go to {args.out}")

    # Tu turno: build the three retrievers over your chunks (both chunking strategies!) and
    # call evaluate_retriever(retriever, gold, args.k) on each. Sweep λ for the hybrid one.
    raise NotImplementedError("Build your retrievers here, then evaluate them on the gold set.")


if __name__ == "__main__":
    main()
