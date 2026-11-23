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

import logging
import os
import re
from pathlib import Path
import threading

from api.schemas import RagResponse, RetrievedChunk
from rag.retriever import Hit

logger = logging.getLogger(__name__)

CHUNKING = os.getenv("ARCA_RAG_CHUNKING", "recursive")  # "recursive" | "semantic"
SEMANTIC_THRESHOLD = 0.75  # chosen with the sweep in rag/evaluate.py (see EXPERIMENTS.md)
CORPUS_DIR = Path("rag/corpus")
INDEX_DIR = Path("rag/index")
GENERATOR_MODEL = os.getenv("ARCA_RLM_BASE_MODEL", "Qwen/Qwen3-4B")
GENERATOR_ADAPTER = os.getenv("ARCA_RLM_ADAPTER") or None
LAMBDA_BM25 = 0.4  # best MRR in the λ sweep with recursive chunking (EXPERIMENTS.md)
MAX_NEW_TOKENS = 1536

RAG_PROMPT_TEMPLATE = """Use the provided up-to-date information to answer the question.
Cite the passages you rely on by writing their id in square brackets, like [{example_id}].
If the information is not enough to answer, say so instead of guessing.

Retrieved information:
{chunks}

Question: {question}
"""

CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9_-]+)\]")

# What /rag answers when the model runs out of tokens while still thinking: there is no answer.
TRUNCATED_ANSWER = "No se pudo completar la respuesta: el razonamiento del modelo se truncó."


def format_chunks(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{hit.chunk.id}] (from {hit.chunk.source})\n{hit.chunk.text}" for hit in hits
    )


def final_answer(raw: str) -> str:
    """The answer of an RLM completion: the ``<answer>`` block, or what follows ``</think>``.

    If the completion ends inside an unclosed ``<think>`` (it hit ``max_new_tokens`` while
    reasoning) there is no answer, and the reasoning must not be returned in its place: ids
    mentioned while thinking would otherwise count as citations.
    """
    from rlm.inference import split_thinking

    if "<think>" in raw and "</think>" not in raw:
        return TRUNCATED_ANSWER
    return split_thinking(raw)[1]


def build_prompt(question: str, hits: list[Hit]) -> str:
    example_id = hits[0].chunk.id if hits else "chunk-id"
    return RAG_PROMPT_TEMPLATE.format(
        example_id=example_id, chunks=format_chunks(hits), question=question
    )


def check_citations(answer: str, valid_ids: set[str]) -> tuple[list[str], list[str]]:
    """Split the ids in square brackets into (valid, invalid), in order of first appearance.

    Valid ids are the ones that exist among the retrieved chunks. An invalid one is a made-up
    id (or any other bracketed text, like ``[1]``): the model cites something it was not given.
    """
    valid: list[str] = []
    invalid: list[str] = []
    for match in CITATION_PATTERN.findall(answer):
        bucket = valid if match in valid_ids else invalid
        if match not in bucket:
            bucket.append(match)
    return valid, invalid


def extract_citations(answer: str, valid_ids: set[str]) -> list[str]:
    """Ids cited in the answer that actually exist among the retrieved chunks, in order."""
    return check_citations(answer, valid_ids)[0]


def to_response(answer: str, hits: list[Hit], retriever: str, model: str = "") -> RagResponse:
    chunks = [
        RetrievedChunk(id=h.chunk.id, source=h.chunk.source, score=h.score, text=h.chunk.text)
        for h in hits
    ]
    citations, invalid = check_citations(answer, {c.id for c in chunks})
    if invalid:
        logger.warning("The answer cites ids that were not retrieved: %s", invalid)
    return RagResponse(
        answer=answer,
        chunks=chunks,
        citations=citations,
        invalid_citations=invalid,
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
        from core.models import get_embedder, get_llm

        from rag.chunking import recursive_chunks, semantic_chunks
        from rag.ingest import load_corpus
        from rag.retriever import build_retrievers

        if CHUNKING not in {"recursive", "semantic"}:
            raise ValueError(f"CHUNKING must be 'recursive' or 'semantic', got {CHUNKING!r}")
        documents = load_corpus(CORPUS_DIR)
        if not documents:
            raise FileNotFoundError(f"No documents in {CORPUS_DIR}: run rag/download_corpus.py")
        embedder = get_embedder()

        if CHUNKING == "recursive":
            chunks = recursive_chunks(documents)
        else:
            chunks = semantic_chunks(
                documents, embedder.embed_passages, distance_threshold=SEMANTIC_THRESHOLD
            )
        self.retrievers = build_retrievers(
            chunks, embedder, persist_dir=INDEX_DIR / CHUNKING, weight_bm25=LAMBDA_BM25
        )

        self.tokenizer, self.model = get_llm(GENERATOR_MODEL, GENERATOR_ADAPTER)
        self.model_id = GENERATOR_MODEL + (f" + {GENERATOR_ADAPTER}" if GENERATOR_ADAPTER else "")

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """Tu turno: dispatch to the dense, bm25 or hybrid retriever."""
        if retriever not in self.retrievers:
            raise ValueError(
                f"retriever must be one of {sorted(self.retrievers)}, got {retriever!r}"
            )
        return self.retrievers[retriever].search(question, top_k=top_k)

    def _generate(self, prompt: str) -> str:
        """Helper: apply the chat template, decode greedily, drop any <think> block.

        With an adapter (our phase-1 RLM) the prompt goes inside the R1-Zero system prompt it
        was trained with, the model thinks first and the answer is the ``<answer>`` block, as
        in ``POST /reasoning``. The base model gets the prompt alone, without thinking.
        """
        import torch

        if GENERATOR_ADAPTER:
            from rlm.data import build_prompt as r1_messages

            text = self.tokenizer.apply_chat_template(
                r1_messages(prompt), tokenize=False, add_generation_prompt=True
            )
        else:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        answer = self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if GENERATOR_ADAPTER:
            return final_answer(answer)
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Tu turno: retrieve -> ``build_prompt`` -> generate -> ``to_response``."""
        hits = self.retrieve(question, top_k, retriever)
        prompt = build_prompt(question, hits)
        answer = self._generate(prompt)
        return to_response(answer, hits, retriever, model=self.model_id)


_pipeline_lock = threading.Lock()
_pipeline: RagPipeline | None = None


def build_pipeline() -> RagPipeline:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            pipeline = RagPipeline()
            pipeline.load()
            _pipeline = pipeline
    return _pipeline


def answer_with_rag(question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_pipeline().answer(question, top_k=top_k, retriever=retriever)
