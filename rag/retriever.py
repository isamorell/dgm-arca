"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

``BM25Retriever`` is complete: it is the keyword baseline you compare against.
``DenseRetriever`` and ``HybridRetriever`` are yours. All three return the same ``Hit``
objects so that ``rag/generate.py`` and ``rag/evaluate.py`` do not care which one is used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


class DenseRetriever:
    """Tu turno: embed the chunks once, store them, embed the query and rank by cosine similarity.

    Use ``rag.embed.Embedder`` for the vectors and Chroma (``chromadb.PersistentClient``)
    or a plain NumPy matrix as the store; persist it under ``rag/index/`` so the API does
    not re-embed the corpus at every start. ``search`` must return ``Hit`` objects with
    the cosine similarity as score.
    """

    name = "dense"

    def __init__(self, chunks: list[Chunk], embedder, persist_dir: str = "rag/index"):
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        raise NotImplementedError


class HybridRetriever:
    """Tu turno: s_hybrid = λ · s_bm25 + (1 - λ) · s_dense, on scores rescaled to [0, 1].

    Take the top ``pool`` hits from each retriever, min-max normalise each score list so
    they are comparable, merge by chunk id, and return the top ``top_k`` by the weighted
    sum. Choose λ with ``rag/evaluate.py`` on your gold set, not by intuition, and keep
    the table.
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        raise NotImplementedError
