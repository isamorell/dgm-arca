"""Data indexing, step 2: chunking.

Two strategies from class. ``recursive_chunks`` is the default of most RAG systems and is
implemented for you with LangChain's splitter. ``semantic_chunks`` is yours to write: the
algorithm is in the docstring and in the slides. Compare both with ``rag/evaluate.py``:
number of chunks, average length, and Recall@k on your gold set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.ingest import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(id=f"{doc.id}-{i:04d}", doc_id=doc.id, source=doc.source, text=piece, index=i)
            )
    return chunks


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter; good enough to feed the semantic chunker."""
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [s.strip() for s in sentences if s and s.strip()]


def semantic_chunks(
    documents: list[Document],
    embed_fn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
) -> list[Chunk]:
    """Tu turno: group consecutive sentences while their embeddings stay close.

    ``embed_fn(list[str]) -> array of shape (n, d)`` with L2-normalised rows (see
    ``rag/embed.py``). The algorithm from the slides:

    1. Split the document into sentences s_1 … s_n.
    2. Embed them: e_1 … e_n.
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars``.

    Plot the distances for one document with the threshold line (as in the slides) and
    include it in your report: it is the best way to justify the threshold you chose.
    """
    raise NotImplementedError


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table."""
    lengths = [len(c.text) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
    }
