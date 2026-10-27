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

CHUNKING = "recursive" # "recursive" | "semantic"
CORPUS_DIR = "rag/corpus"
GENERATOR_MODEL = "Qwen/Qwen3-1.7B"
LAMBDA_BM25 = 0.5

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
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rag.chunking import recursive_chunks, semantic_chunks
        from rag.embed import Embedder
        from rag.ingest import load_corpus
        from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

        documents = load_corpus(CORPUS_DIR)
        embedder = Embedder()

        if CHUNKING == "recursive":
            chunks = recursive_chunks(documents)
        else:
            chunks = semantic_chunks(documents, embedder.embed_passages)

        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, persist_dir=f"rag/index/{CHUNKING}")
        hybrid = HybridRetriever(bm25, dense, weight_bm25=LAMBDA_BM25)
        self.retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}

        self.model_id = os.environ.get("ARCA_RLM_BASE_MODEL")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )

        self.model.eval()
        

    def retrieve(self, question: str, top_k: int, retriever: str) -> list[Hit]:
        """Tu turno: dispatch to the dense, bm25 or hybrid retriever."""
        if retriever not in self.retrievers.keys():
            raise ValueError(f"Retriever must be one of {self.retrievers.keys()}, got {retriever!r}")
        return self.retrievers[retriever].search(question, top_k=top_k)

    def _generate(self, prompt: str) -> str:
        """Helper: apply the chat template, decode greedily, drop any <think> block."""
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
            out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

    def answer(self, question: str, top_k: int = 5, retriever: str = "hybrid") -> RagResponse:
        """Tu turno: retrieve -> ``build_prompt`` -> generate -> ``to_response``."""
        hits = self.retrieve(question, top_k, retriever)
        prompt = build_prompt(question, hits)
        answer = self._generate(prompt)
        return to_response(answer, hits, retriever, model=self.model_id)


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
