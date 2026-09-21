"""Retrieval and generation: augment the prompt with the retrieved chunks and answer with citations.

``answer_with_rag`` is what ``POST /rag`` calls. It must:

1. retrieve ``top_k`` chunks with the requested retriever (dense, bm25 or hybrid);
2. build the augmented prompt (``RAG_PROMPT_TEMPLATE`` below is the one from class, with
   chunk ids so the model can cite them);
3. generate the answer with your model;
4. extract the citations and return everything in a ``RagResponse``.

A grader will check that every cited id exists among the returned chunks, and will read
the answers to see whether they actually rest on the retrieved text.
"""

from __future__ import annotations

import re

from api.schemas import RagResponse, RetrievedChunk
from rag.retriever import Hit

RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their id in square brackets, like [{example_id}].
If the information is not enough to answer, say so instead of guessing.

Retrieved information:
{chunks}

Question: {question}
"""

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}" for hit in hits
    )


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    return RAG_PROMPT_TEMPLATE.format(
        example_id=example_id, chunks=format_chunks(hits), question=question
    )


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    seen: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        if match in valid_ids and match not in seen:
            seen.append(match)
    return seen


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=extract_citations(answer, {c.id for c in chunks}),
        retriever=retriever,
        model=model,
    )


class RagPipeline:
    """Everything ``POST /rag`` needs, built once and reused."""

    def load(self) -> None:
        """Tu turno: ingest -> chunk -> build the three retrievers -> load the generator model.

        Persist the dense index under ``rag/index/`` so this is fast after the first run.
        The generator can be your phase-1 RLM or the base model; record which in ``model``.
        """
        raise NotImplementedError("Phase 3 is not implemented yet: see rag/README.md.")

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """Tu turno: dispatch to the dense, bm25 or hybrid retriever."""
        raise NotImplementedError

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Tu turno: retrieve -> ``build_prompt`` -> generate -> ``to_response``."""
        raise NotImplementedError


_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    if _pipeline is None:
        pipeline = RagPipeline()
        pipeline.load()
        _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)
