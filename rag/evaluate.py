"""Phase 3 evaluation: Recall@k and MRR of each retriever on your gold set.

The gold set is a JSONL file (``rag/gold/example_gold.jsonl`` shows the format) with a
question and the id(s) of the chunk(s) that answer it. Fifty questions is the minimum;
write them yourselves while reading the corpus, and note the ones where two chunks are
equally valid: they will tell you a lot about your chunking.

Run::

    uv run python -m rag.evaluate --gold rag/gold/gold.jsonl --k 5

To choose the threshold of the semantic chunking with retrieval numbers, sweep it::

    uv run python -m rag.evaluate --gold rag/gold/gold.jsonl --thresholds 0.6 0.7 0.75 0.8 0.85
"""

from __future__ import annotations

import argparse
import json
import re
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


_MD_HEADING = re.compile(r"^#{1,6}[ \t]+", flags=re.MULTILINE)


def _norm(text: str) -> str:
    """Collapse whitespace and drop Markdown heading marks.

    Chunks carry their heading without the ``##`` (see ``add_headers`` in ``rag.chunking``),
    so evidence copied from the raw file with its heading line must lose them to match.
    """
    return re.sub(r"\s+", " ", _MD_HEADING.sub("", text)).strip().lower()


def resolve_gold(gold: list[dict], chunks) -> list[dict]:
    """Recompute ``relevant_chunk_ids`` for a given chunking.

    Chunk ids change with the chunking strategy, so a gold set annotated with the ids of one
    strategy is useless for the other. If an item has an ``evidence`` field (a verbatim
    passage, or a list of them, copied from the corpus), the relevant chunks are the ones
    whose text contains it. Items without ``evidence`` keep their ``relevant_chunk_ids``.
    """
    normalized = [(c.id, _norm(c.text)) for c in chunks]
    resolved = []
    for item in gold:
        evidence = item.get("evidence")
        if not evidence:
            resolved.append(item)
            continue
        pieces = [_norm(p) for p in ([evidence] if isinstance(evidence, str) else evidence)]
        ids = [cid for cid, text in normalized if any(p in text for p in pieces)]
        resolved.append({**item, "relevant_chunk_ids": ids})
    return resolved


def failed_questions(retriever, gold: list[dict], k: int) -> list[dict]:
    """Questions whose relevant chunk is not in the top k: material for your interpretation."""
    failures = []
    for item in gold:
        ranked = [hit.chunk.id for hit in retriever.search(item["question"], top_k=k)]
        if not set(ranked) & set(item["relevant_chunk_ids"]):
            failures.append(
                {
                    "question": item["question"],
                    "relevant": item["relevant_chunk_ids"],
                    "retrieved": ranked,
                }
            )
    return failures


def over_all(metrics: dict[str, float], n_total: int, k: int) -> dict[str, float]:
    """Rescale metrics computed on the resolved questions to the whole gold set.

    A question whose evidence fits in no chunk cannot be retrieved, so it counts as 0.
    Without this, a chunking that splits the evidence would be scored on easier questions.
    """
    factor = metrics["n_questions"] / max(n_total, 1)
    return {f"recall@{k}": metrics[f"recall@{k}"] * factor, "mrr": metrics["mrr"] * factor}


def threshold_sweep(
    documents, embedder, gold: list[dict], thresholds: list[float], k: int, lam: float
) -> dict[str, dict]:
    """Semantic chunking at every threshold: chunk stats and the three retrievers."""
    from rag.chunking import cached_embed, describe, semantic_chunks
    from rag.retriever import build_retrievers

    embed_fn = cached_embed(embedder.embed_passages)  # sentences are embedded only once
    sweep: dict[str, dict] = {}
    for threshold in thresholds:
        chunks = semantic_chunks(documents, embed_fn, distance_threshold=threshold)
        items = resolve_gold(gold, chunks)
        usable = [it for it in items if it["relevant_chunk_ids"]]
        retrievers = build_retrievers(chunks, embedder, persist_dir=None, weight_bm25=lam)
        entry = {
            "stats": describe(chunks),
            "n_unresolved": len(items) - len(usable),
            "retrievers": {},
        }
        for rname, retriever in retrievers.items():
            metrics = evaluate_retriever(retriever, usable, k)
            entry["retrievers"][rname] = {**metrics, "all": over_all(metrics, len(gold), k)}
        sweep[str(threshold)] = entry
    return sweep


def print_threshold_sweep(sweep: dict[str, dict], k: int) -> None:
    key = f"recall@{k}"
    print(
        f"\nBarrido del umbral del chunking semántico (métricas sobre las {key} resueltas / todas):"
    )
    print(
        f"{'umbral':>6} {'chunks':>7} {'media':>6} {'sin res.':>8}  "
        + "  ".join(f"{r + ' ' + key:>18} {r + ' mrr':>14}" for r in ("bm25", "dense", "hybrid"))
    )
    for threshold, entry in sweep.items():
        s = entry["stats"]
        cells = []
        for m in entry["retrievers"].values():
            cells.append(
                f"{m[key]:>8.3f} / {m['all'][key]:.3f} {m['mrr']:>6.3f} / {m['all']['mrr']:.3f}"
            )
        print(
            f"{threshold:>6} {s['n_chunks']:>7} {s['mean_chars']:>6.0f} "
            f"{entry['n_unresolved']:>8}  " + "  ".join(cells)
        )


def print_report(report: dict, k: int) -> None:
    key = f"recall@{k}"
    for name, entry in report["chunking"].items():
        s = entry["stats"]
        print(f"\n=== chunking: {name} ===")
        print(
            f"{s['n_chunks']} chunks | mean {s['mean_chars']:.0f} chars "
            f"(min {s['min_chars']}, max {s['max_chars']}) | "
            f"{entry['n_unresolved']} gold questions without a matching chunk"
        )
        print(
            f"{'retriever':<10} {key:>10} {'mrr':>8} "
            f"{key + ' (todas)':>18} {'mrr (todas)':>12} {'fallos':>8}"
        )
        for rname, m in entry["retrievers"].items():
            n_fail = len(entry["failures"][rname])
            print(
                f"{rname:<10} {m[key]:>10.3f} {m['mrr']:>8.3f} "
                f"{m['all'][key]:>18.3f} {m['all']['mrr']:>12.3f} {n_fail:>8}"
            )
        print(f"\nBarrido de λ (peso de BM25) en hybrid, chunking {name}:")
        print(f"{'λ':>5} {key:>10} {'mrr':>8}")
        for lam, m in entry["lambda_sweep"].items():
            print(f"{lam:>5} {m[key]:>10.3f} {m['mrr']:>8.3f}")
        best = max(entry["lambda_sweep"].items(), key=lambda kv: (kv[1]["mrr"], kv[1][key]))
        print(f"Mejor λ por MRR: {best[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/example_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="reports/phase3_eval.json")
    parser.add_argument("--corpus", default="rag/corpus")
    parser.add_argument("--index-dir", default="rag/index")
    parser.add_argument("--threshold", type=float, default=0.75, help="semantic chunking")
    parser.add_argument("--lam", type=float, default=0.5, help="λ of the hybrid in the main table")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=None,
        help="only sweep the semantic threshold over these values (no main table)",
    )
    args = parser.parse_args()
    gold = load_gold(args.gold)
    print(f"{len(gold)} gold questions loaded; results will go to {args.out}")

    # Tu turno: build the three retrievers over your chunks (both chunking strategies!) and
    # call evaluate_retriever(retriever, gold, args.k) on each. Sweep λ for the hybrid one.
    from rag.chunking import describe, recursive_chunks, semantic_chunks
    from rag.embed import Embedder
    from rag.ingest import load_corpus
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    documents = load_corpus(Path(args.corpus))
    embedder = Embedder()

    if args.thresholds:
        sweep = threshold_sweep(documents, embedder, gold, args.thresholds, args.k, args.lam)
        print_threshold_sweep(sweep, args.k)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"k": args.k, "gold": args.gold, "lam": args.lam, "threshold_sweep": sweep}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved to {out}")
        return
    strategies = {
        "recursive": recursive_chunks(documents),
        "semantic": semantic_chunks(
            documents, embedder.embed_passages, distance_threshold=args.threshold
        ),
    }
    lambdas = [round(i / 10, 1) for i in range(11)]

    report: dict = {"k": args.k, "gold": args.gold, "threshold": args.threshold, "chunking": {}}

    for name, chunks in strategies.items():
        items = resolve_gold(gold, chunks)
        usable = [it for it in items if it["relevant_chunk_ids"]]

        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, persist_dir=f"{args.index_dir}/{name}")
        hybrid = HybridRetriever(bm25, dense, weight_bm25=args.lam)
        retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

        entry: dict = {
            "stats": describe(chunks),
            "n_unresolved": len(items) - len(usable),
            "retrievers": {},
            "failures": {},
            "lambda_sweep": {},
        }

        for rname, retriever in retrievers.items():
            metrics = evaluate_retriever(retriever, usable, args.k)
            entry["retrievers"][rname] = {**metrics, "all": over_all(metrics, len(gold), args.k)}
            entry["failures"][rname] = failed_questions(retriever, usable, args.k)

        for lam in lambdas:
            hybrid.weight_bm25 = lam
            entry["lambda_sweep"][str(lam)] = evaluate_retriever(hybrid, usable, args.k)
        hybrid.weight_bm25 = args.lam
        report["chunking"][name] = entry

    print_report(report, args.k)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
