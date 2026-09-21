"""Retrieval metrics and the citation extractor are given; these tests document them."""

import pytest

from rag.evaluate import recall_at_k, reciprocal_rank
from rag.generate import extract_citations


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["x"], {"a"}, k=5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "b"], {"a"}) == 0.5
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_extract_citations_keeps_only_valid_ids_in_order():
    answer = "The axolotl regrows limbs [doc1-0001]. See also [doc1-0001] and [ghost-9999]."
    assert extract_citations(answer, {"doc1-0001", "doc1-0002"}) == ["doc1-0001"]


def test_recursive_chunking_respects_size():
    pytest.importorskip("langchain_text_splitters")
    from rag.chunking import describe, recursive_chunks
    from rag.ingest import Document

    doc = Document(id="d1", source="notes.md", text=("Sentence one. " * 40 + "\n\n") * 5)
    chunks = recursive_chunks([doc], chunk_size=200, chunk_overlap=20)
    stats = describe(chunks)
    assert stats["n_chunks"] > 5
    assert stats["max_chars"] <= 200
    assert chunks[0].id == "d1-0000"
