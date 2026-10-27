"""Retrieval: similarity search, keyword search (BM25) and the hybrid of both.

All three retrievers return the same ``Hit`` objects, so ``rag/generate.py`` and
``rag/evaluate.py`` do not care which one is used:

* ``BM25Retriever``: keyword ranking, the lexical baseline.
* ``DenseRetriever``: instruction-aware embeddings and cosine similarity over a NumPy matrix
  that is saved to disk, so the corpus is embedded only once.
* ``HybridRetriever``: ``λ · bm25 + (1 - λ) · dense`` over min-max normalised scores.
"""

from __future__ import annotations

import hashlib
import os
import re
import unicodedata
import warnings
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk


@dataclass
class Hit:
    chunk: Chunk
    score: float


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, without accents and with ``0,9`` written as ``0.9``.

    Accents are dropped because students type "perfusion" as often as "perfusión", and the
    decimal comma is unified so that "0,9 %" in a data sheet matches "0.9" in a question.
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    return re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+", text)


class BM25Retriever:
    """Keyword ranking: term frequency, inverse document frequency, length normalisation."""

    name = "bm25"

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c.text) for c in chunks]) if chunks else None

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if self.index is None:
            return []
        scores = self.index.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def _fingerprint(chunks: list[Chunk], model_id: str) -> str:
    """Identifies what the saved vectors were computed from: model and exact chunk texts."""
    digest = hashlib.sha256(model_id.encode("utf-8"))
    for chunk in chunks:
        digest.update(b"\x1e" + chunk.id.encode("utf-8") + b"\x1f" + chunk.text.encode("utf-8"))
    return digest.hexdigest()


class DenseRetriever:
    """Embed the chunks once, store them, embed the query and rank by cosine similarity.

    Passages are embedded without instruction and the query with ``query_instruction`` (the
    default of ``rag.embed`` when ``None``). Since passage vectors do not depend on the
    instruction, you can change ``query_instruction`` on a built retriever and compare
    instructions without re-embedding the corpus.

    The vectors are saved in ``persist_dir`` (``rag/index/``, outside git) as
    ``dense_<fingerprint>.npz``. The fingerprint depends on the embedding model and on the
    exact text of every chunk, so a different chunking strategy, threshold or corpus gets its
    own file and a stale index is never loaded. ``persist_dir=None`` keeps everything in
    memory. With a few hundred chunks an exact matrix product is instantaneous and the score
    is exactly the cosine similarity, so no approximate index is needed.
    """

    name = "dense"

    def __init__(
        self,
        chunks: list[Chunk],
        embedder,
        persist_dir: str | Path | None = "rag/index",
        query_instruction: str | None = None,
        batch_size: int = 32,
        query_cache_size: int = 512,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.persist_dir = Path(persist_dir) if persist_dir is not None else None
        self.fingerprint = _fingerprint(chunks, getattr(embedder, "model_id", "unknown"))
        self._query_cache: OrderedDict[tuple[str | None, str], np.ndarray] = OrderedDict()
        self._query_cache_size = query_cache_size
        self.matrix = self._load_or_build()

    @property
    def index_path(self) -> Path | None:
        if self.persist_dir is None:
            return None
        return self.persist_dir / f"dense_{self.fingerprint[:16]}.npz"

    def _load_or_build(self) -> np.ndarray:
        path = self.index_path
        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as data:
                    matrix = data["embeddings"]
                    same_chunks = str(data["fingerprint"]) == self.fingerprint
                    if same_chunks and len(matrix) == len(self.chunks):
                        return matrix
            except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
                pass  # unreadable or incomplete file: rebuild it below
        if not self.chunks:
            return np.zeros((0, 0), dtype=np.float32)
        matrix = np.asarray(
            self.embedder.embed_passages([c.text for c in self.chunks], batch_size=self.batch_size),
            dtype=np.float32,
        )
        if path is not None:
            self._save(path, matrix)
        return matrix

    def _save(self, path: Path, matrix: np.ndarray) -> None:
        """Write to a temporary file and rename, so a crash never leaves a half-written index."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.stem + ".tmp.npz")
            np.savez(tmp, embeddings=matrix, fingerprint=np.array(self.fingerprint))
            os.replace(tmp, path)
        except OSError as error:
            warnings.warn(f"Could not save the dense index to {path}: {error}", stacklevel=2)

    def _embed_query(self, query: str) -> np.ndarray:
        key = (self.query_instruction, query)
        vector = self._query_cache.get(key)
        if vector is None:
            kwargs = {}
            if self.query_instruction is not None:
                kwargs["instruction"] = self.query_instruction
            embedded = self.embedder.embed_queries([query], **kwargs)
            vector = np.asarray(embedded, dtype=np.float32)[0]
            self._query_cache[key] = vector
            while len(self._query_cache) > self._query_cache_size:
                self._query_cache.popitem(last=False)
        else:
            self._query_cache.move_to_end(key)
        return vector

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        if not self.chunks or top_k <= 0:
            return []
        scores = self.matrix @ self._embed_query(query)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [Hit(self.chunks[i], float(scores[i])) for i in order]


def _minmax(scores: list[float]) -> list[float]:
    """Rescale to [0, 1]. A list without spread is 1 if it carries signal (> 0) and 0 if not."""
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if high - low < 1e-12:
        return [1.0 if high > 0 else 0.0] * len(scores)
    return [(s - low) / (high - low) for s in scores]


class HybridRetriever:
    """s_hybrid = λ · s_bm25 + (1 - λ) · s_dense, on scores rescaled to [0, 1].

    Take the top ``pool`` hits from each retriever, min-max normalise each score list so
    they are comparable, merge by chunk id, and return the top ``top_k`` by the weighted
    sum. A chunk found by only one retriever gets 0 from the other. ``weight_bm25`` is λ
    and can be changed on a built retriever, which is what the λ sweep in
    ``rag/evaluate.py`` does: λ = 1 is BM25 alone and λ = 0 is the dense retriever alone.
    The returned score is the weighted sum, always in [0, 1].
    """

    name = "hybrid"

    def __init__(
        self, bm25: BM25Retriever, dense: DenseRetriever, weight_bm25: float = 0.5, pool: int = 20
    ):
        self.bm25, self.dense, self.weight_bm25, self.pool = bm25, dense, weight_bm25, pool

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        weight = self.weight_bm25
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"weight_bm25 must be in [0, 1], got {weight}")
        pool = max(self.pool, top_k)
        dense_hits = self.dense.search(query, pool)
        bm25_hits = self.bm25.search(query, pool)

        # chunk id -> [chunk, normalised bm25 score, normalised dense score]
        merged: dict[str, list] = {}
        for hit, score in zip(dense_hits, _minmax([h.score for h in dense_hits]), strict=True):
            merged[hit.chunk.id] = [hit.chunk, 0.0, score]
        for hit, score in zip(bm25_hits, _minmax([h.score for h in bm25_hits]), strict=True):
            merged.setdefault(hit.chunk.id, [hit.chunk, 0.0, 0.0])[1] = score

        ranked = [
            Hit(chunk, weight * bm25 + (1.0 - weight) * dense)
            for chunk, bm25, dense in merged.values()
        ]
        ranked.sort(key=lambda hit: hit.score, reverse=True)  # stable: ties keep dense order
        return ranked[:top_k]


def build_retrievers(
    chunks: list[Chunk],
    embedder,
    persist_dir: str | Path | None = "rag/index",
    weight_bm25: float = 0.5,
    pool: int = 20,
    query_instruction: str | None = None,
) -> dict[str, BM25Retriever | DenseRetriever | HybridRetriever]:
    """The three retrievers over the same chunks, keyed by the ``retriever`` name of ``/rag``."""
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(
        chunks, embedder, persist_dir=persist_dir, query_instruction=query_instruction
    )
    hybrid = HybridRetriever(bm25, dense, weight_bm25=weight_bm25, pool=pool)
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}
