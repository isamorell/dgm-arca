"""Data indexing, step 3: embeddings.

We use Qwen3-Embedding through Sentence Transformers. Remember the point of the lecture:
relevance depends on the task, so the *query* is embedded together with an instruction
that says what "relevant" means, while passages are embedded as they are. Change the
instruction and you change which chunks come back; that is a lever you should play with.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a user question, retrieve the passages from the knowledge base that answer it."
)


class Embedder:
    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_id
        self.model = SentenceTransformer(model_id, device=device)

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Passages are embedded without instruction. Rows are L2-normalised."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )

    def embed_queries(
        self, texts: list[str], instruction: str = DEFAULT_QUERY_INSTRUCTION
    ) -> np.ndarray:
        """Queries are prefixed with the task instruction (instruction-aware embeddings)."""
        prompt = f"Instruct: {instruction}\nQuery: "
        return self.model.encode(texts, prompt=prompt, normalize_embeddings=True)


def cosine_similarity(queries: np.ndarray, passages: np.ndarray) -> np.ndarray:
    """(n_queries, n_passages) similarities for normalised embeddings: just a dot product."""
    return queries @ passages.T
