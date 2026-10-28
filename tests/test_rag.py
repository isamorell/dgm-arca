"""Retrieval metrics and the citation extractor are given; these tests document them.

The chunking strategies and the three retrievers are tested without any model: a fake
embedder (bag of words) and one-hot "topic" embeddings stand in for Qwen3-Embedding.
"""

import numpy as np
import pytest

from rag.chunking import (
    Chunk,
    cached_embed,
    plot_distances,
    recursive_chunks,
    semantic_chunks,
    sentence_distances,
    split_sections,
    split_sentences,
)
from rag.embed import DEFAULT_QUERY_INSTRUCTION
from rag.evaluate import recall_at_k, reciprocal_rank
from rag.generate import extract_citations
from rag.retriever import (
    BM25Retriever,
    DenseRetriever,
    Hit,
    HybridRetriever,
    build_retrievers,
    tokenize,
)


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
    from rag.chunking import describe
    from rag.ingest import Document

    doc = Document(id="d1", source="notes.md", text=("Sentence one. " * 40 + "\n\n") * 5)
    chunks = recursive_chunks([doc], chunk_size=200, chunk_overlap=20)
    stats = describe(chunks)
    assert stats["n_chunks"] > 5
    assert stats["max_chars"] <= 200
    assert chunks[0].id == "d1-0000"


# ----------------------------------------------------------------------------- helpers


def _doc(text, doc_id="d1", source="d1.md"):
    from rag.ingest import Document

    return Document(id=doc_id, source=source, text=text)


def _topic_embed(sentences):
    """One-hot by topic: same topic -> distance 0, different topic -> distance 1."""
    out = np.zeros((len(sentences), 3), dtype=np.float32)
    for i, sentence in enumerate(sentences):
        if "amoxicilina" in sentence:
            out[i, 0] = 1.0
        elif "insulina" in sentence:
            out[i, 1] = 1.0
        else:
            out[i, 2] = 1.0
    return out


class FakeEmbedder:
    """Bag-of-words vectors (no model, deterministic) with the interface of rag.embed.Embedder."""

    model_id = "fake-embedder"
    DIM = 256

    def __init__(self, vocab=None):
        self.vocab = {} if vocab is None else vocab
        self.passage_calls = 0
        self.instructions = []

    def _vector(self, text):
        from rag.retriever import tokenize

        vector = np.zeros(self.DIM, dtype=np.float32)
        for token in tokenize(text):
            vector[self.vocab.setdefault(token, len(self.vocab))] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def embed_passages(self, texts, batch_size=32):
        self.passage_calls += 1
        return np.stack([self._vector(t) for t in texts])

    def embed_queries(self, texts, instruction=DEFAULT_QUERY_INSTRUCTION):
        self.instructions.append(instruction)
        return np.stack([self._vector(t) for t in texts])


def _chunk(chunk_id, text):
    return Chunk(id=chunk_id, doc_id="d", source="s.md", text=text, index=0)


CORPUS = {
    "c0": "amoxicilina clavulanico se reconstituye en agua para inyectables",
    "c1": "insulina rapida se conserva en nevera entre dos y ocho grados",
    "c2": "vancomicina se perfunde lentamente en una hora para evitar reacciones",
    "c3": "heparina sodica se administra por via subcutanea o intravenosa",
    "c4": "potasio cloruro nunca se administra en bolo directo",
    "c5": "morfina se titula segun dolor y funcion respiratoria",
}


def _chunks():
    return [_chunk(k, v) for k, v in CORPUS.items()]


class StubRetriever:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k=5):
        return self._hits[:top_k]


def _hits(**scores):
    return [Hit(_chunk(k, k), v) for k, v in scores.items()]


# ------------------------------------------------------------------------ tokenisation


def test_tokenize_folds_accents_and_decimal_comma():
    tokens = tokenize("Ceftriaxona 0,9 % — Perfusión IV; Máx. 2 g/día")
    assert tokens == ["ceftriaxona", "0.9", "perfusion", "iv", "max", "2", "g", "dia"]


# --------------------------------------------------------------------- sections, sentences


def test_split_sections_separates_headings_and_drops_empty_ones():
    text = "# T\n- a\n\n## S1\ntexto uno\n\n## S2\n\ntexto dos\n## Vacia\n"
    assert split_sections(text) == [("T", "- a"), ("S1", "texto uno"), ("S2", "texto dos")]
    assert split_sections("sin encabezados") == [("", "sin encabezados")]
    assert split_sections("   ") == []


def test_split_sentences_keeps_abbreviations_and_decimals_together():
    text = "Administrar p. ej. por vía IV. Diluir a 0,9 %. Dosis máx. 5 mg.\nSegunda línea"
    assert split_sentences(text) == [
        "Administrar p. ej. por vía IV.",
        "Diluir a 0,9 %.",
        "Dosis máx. 5 mg.",
        "Segunda línea",
    ]


# --------------------------------------------------------------------------- recursive


def test_recursive_chunks_follow_sections_and_prepend_the_header():
    pytest.importorskip("langchain_text_splitters")
    doc = _doc(
        "# Med X\n- dato\n\n"
        "## Med X — 4.2 Posología\nDosis uno.\n\n"
        "## Med X — 6.3 Validez\n24 horas."
    )
    chunks = recursive_chunks([doc])
    headings = ["Med X", "Med X — 4.2 Posología", "Med X — 6.3 Validez"]
    assert [c.heading for c in chunks] == headings
    assert chunks[2].text == "Med X — 6.3 Validez\n24 horas."
    assert [c.id for c in chunks] == ["d1-0000", "d1-0001", "d1-0002"]

    plain = recursive_chunks([doc], by_section=False, add_headers=False)
    assert len(plain) == 1  # the three sections fit in one chunk when structure is ignored
    assert "24 horas." in plain[0].text

    no_header = recursive_chunks([doc], add_headers=False)
    assert no_header[2].text == "24 horas."


def test_recursive_chunks_count_the_header_in_the_size_limit():
    pytest.importorskip("langchain_text_splitters")
    heading = "Medicamento X — 4.2 Posología"
    doc = _doc(f"## {heading}\n" + "Dosis de prueba. " * 30)
    chunks = recursive_chunks([doc], chunk_size=120, chunk_overlap=10)
    assert len(chunks) > 3
    assert all(len(c.text) <= 120 for c in chunks)
    assert all(c.text.startswith(heading + "\n") for c in chunks)
    # cuts fall after the full stop: no chunk starts with leftover punctuation
    assert all(not c.text.split("\n", 1)[1].startswith((".", " ")) for c in chunks)


# ---------------------------------------------------------------------------- semantic


def test_semantic_chunks_cut_where_the_topic_changes():
    text = (
        "La amoxicilina se diluye. La amoxicilina se reconstituye. "
        "La insulina se refrigera. La insulina se agita."
    )
    chunks = semantic_chunks([_doc(text)], _topic_embed)
    assert [c.text for c in chunks] == [
        "La amoxicilina se diluye. La amoxicilina se reconstituye.",
        "La insulina se refrigera. La insulina se agita.",
    ]
    assert [c.id for c in chunks] == ["d1-0000", "d1-0001"]
    assert [c.index for c in chunks] == [0, 1]
    assert all(c.text in text for c in chunks)  # verbatim: evidence matching keeps working


def test_semantic_chunks_never_cross_a_section_and_add_the_header():
    text = "## A\nLa amoxicilina se diluye.\n\n## B\nLa amoxicilina se reconstituye."
    chunks = semantic_chunks([_doc(text)], _topic_embed)
    assert [c.text for c in chunks] == [
        "A\nLa amoxicilina se diluye.",
        "B\nLa amoxicilina se reconstituye.",
    ]
    assert [c.text for c in semantic_chunks([_doc(text)], _topic_embed, add_headers=False)] == [
        "La amoxicilina se diluye.",
        "La amoxicilina se reconstituye.",
    ]


def test_semantic_chunks_respect_max_chunk_chars():
    text = " ".join(f"La amoxicilina se diluye {n}." for n in "abcdefgh")
    chunks = semantic_chunks([_doc(text)], _topic_embed, max_chunk_chars=60)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 60 for c in chunks)


def test_semantic_chunks_split_one_very_long_sentence():
    text = "La amoxicilina " + "se diluye " * 40  # a single sentence of ~400 characters
    chunks = semantic_chunks([_doc(text)], _topic_embed, max_chunk_chars=100)
    assert len(chunks) >= 4
    assert all(len(c.text) <= 100 for c in chunks)


def test_semantic_chunks_min_chunk_chars_merges_tiny_chunks():
    text = (
        "La amoxicilina se diluye. La insulina se refrigera. "
        "La insulina se agita. La insulina se guarda."
    )
    assert len(semantic_chunks([_doc(text)], _topic_embed)) == 2
    assert len(semantic_chunks([_doc(text)], _topic_embed, min_chunk_chars=100)) == 1


def test_semantic_chunks_percentile_threshold():
    text = "La amoxicilina se diluye. La amoxicilina se agita. La insulina se refrigera."
    # distances are [0, 1]: the 100th percentile is 1 (nothing is above it), the 0th is 0
    assert len(semantic_chunks([_doc(text)], _topic_embed, percentile=100)) == 1
    assert len(semantic_chunks([_doc(text)], _topic_embed, percentile=0)) == 2


def test_semantic_chunks_edge_cases():
    assert semantic_chunks([], _topic_embed) == []
    assert semantic_chunks([_doc("   \n ")], _topic_embed) == []
    only = semantic_chunks([_doc("Una sola frase.")], _topic_embed)
    assert [c.text for c in only] == ["Una sola frase."]


def test_semantic_chunks_embed_the_whole_corpus_in_one_call():
    calls = []

    def embed(sentences):
        calls.append(len(sentences))
        return _topic_embed(sentences)

    docs = [_doc("La insulina se agita. La insulina se guarda.", "a"), _doc("Otra frase.", "b")]
    chunks = semantic_chunks(docs, embed)
    assert calls == [3]
    assert [c.id for c in chunks] == ["a-0000", "b-0000"]


def test_cached_embed_embeds_each_text_once():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return _topic_embed(texts)

    cached = cached_embed(embed)
    first = cached(["La insulina.", "La amoxicilina.", "La insulina."])
    second = cached(["La amoxicilina.", "Nueva."])
    assert calls == [["La insulina.", "La amoxicilina."], ["Nueva."]]
    assert first.shape == (3, 3) and second.shape == (2, 3)
    assert np.array_equal(first[0], first[2])


def test_sentence_distances():
    sentences = ["La amoxicilina uno.", "La amoxicilina dos.", "La insulina."]
    assert sentence_distances(sentences, _topic_embed).tolist() == [0.0, 1.0]
    assert len(sentence_distances(["sola"], _topic_embed)) == 0


def test_plot_distances_writes_a_png(tmp_path):
    pytest.importorskip("matplotlib")
    doc = _doc("## A\nLa amoxicilina uno. La insulina dos.\n\n## B\nLa insulina tres. Otra cosa.")
    out = tmp_path / "plots" / "distances.png"
    distances = plot_distances(doc, _topic_embed, out, thresholds=(0.25, 0.5))
    assert out.exists() and out.stat().st_size > 0
    assert len(distances) == 3


# ------------------------------------------------------------------------------- dense


def test_dense_retriever_ranks_by_cosine_similarity():
    dense = DenseRetriever(_chunks(), FakeEmbedder(), persist_dir=None)
    hits = dense.search("conservar insulina nevera", top_k=3)
    assert [h.chunk.id for h in hits][0] == "c1"
    assert len(hits) == 3
    assert hits[0].score == pytest.approx(2 / np.sqrt(3 * 11), abs=1e-5)  # insulina, nevera
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert len(dense.search("insulina", top_k=50)) == 6


def test_dense_retriever_without_chunks_returns_nothing():
    assert DenseRetriever([], FakeEmbedder(), persist_dir=None).search("insulina") == []


def test_dense_index_is_persisted_and_reused(tmp_path):
    vocab = {}
    first = FakeEmbedder(vocab)
    DenseRetriever(_chunks(), first, persist_dir=tmp_path)
    assert first.passage_calls == 1

    second = FakeEmbedder(vocab)
    reloaded = DenseRetriever(_chunks(), second, persist_dir=tmp_path)
    assert second.passage_calls == 0  # loaded from disk
    assert reloaded.search("insulina nevera", top_k=1)[0].chunk.id == "c1"


def test_dense_index_is_rebuilt_when_the_chunks_change(tmp_path):
    vocab = {}
    DenseRetriever(_chunks(), FakeEmbedder(vocab), persist_dir=tmp_path)
    changed = _chunks()
    changed[1] = _chunk("c1", "insulina rapida se guarda en el frigorifico")
    embedder = FakeEmbedder(vocab)
    DenseRetriever(changed, embedder, persist_dir=tmp_path)
    assert embedder.passage_calls == 1
    assert len(list(tmp_path.glob("dense_*.npz"))) == 2  # both versions coexist


def test_dense_index_ignores_a_corrupt_file(tmp_path):
    vocab = {}
    dense = DenseRetriever(_chunks(), FakeEmbedder(vocab), persist_dir=tmp_path)
    dense.index_path.write_bytes(b"not an npz")
    embedder = FakeEmbedder(vocab)
    DenseRetriever(_chunks(), embedder, persist_dir=tmp_path)
    assert embedder.passage_calls == 1


def test_dense_query_instruction_is_a_parameter_and_queries_are_cached():
    embedder = FakeEmbedder()
    dense = DenseRetriever(_chunks(), embedder, persist_dir=None)
    dense.search("insulina")
    dense.search("insulina")  # cached
    dense.query_instruction = "Otra instruccion"
    dense.search("insulina")
    assert embedder.instructions == [DEFAULT_QUERY_INSTRUCTION, "Otra instruccion"]


# ------------------------------------------------------------------------------ hybrid


def _stub_hybrid(weight):
    bm25 = StubRetriever(_hits(A=10.0, B=5.0, C=0.0))
    dense = StubRetriever(_hits(C=0.9, B=0.8, D=0.1))
    return HybridRetriever(bm25, dense, weight_bm25=weight)


def test_hybrid_extremes_reproduce_each_retriever():
    only_bm25 = _stub_hybrid(1.0).search("q", top_k=2)
    assert [h.chunk.id for h in only_bm25] == ["A", "B"]
    only_dense = _stub_hybrid(0.0).search("q", top_k=2)
    assert [h.chunk.id for h in only_dense] == ["C", "B"]


def test_hybrid_merges_by_id_with_normalised_scores():
    hits = _stub_hybrid(0.5).search("q", top_k=10)
    ids = [h.chunk.id for h in hits]
    assert sorted(ids) == ["A", "B", "C", "D"]  # B appears once although both lists have it
    assert ids[0] == "B"
    assert hits[0].score == pytest.approx(0.5 * 0.5 + 0.5 * (0.8 - 0.1) / (0.9 - 0.1))
    assert all(0.0 <= h.score <= 1.0 for h in hits)
    assert len(_stub_hybrid(0.5).search("q", top_k=2)) == 2


def test_hybrid_weight_can_be_changed_after_building():
    hybrid = _stub_hybrid(1.0)
    assert hybrid.search("q", top_k=1)[0].chunk.id == "A"
    hybrid.weight_bm25 = 0.0
    assert hybrid.search("q", top_k=1)[0].chunk.id == "C"


def test_hybrid_survives_a_query_bm25_cannot_match():
    bm25 = StubRetriever(_hits(A=0.0, B=0.0))
    dense = StubRetriever(_hits(B=0.7, C=0.2))
    hits = HybridRetriever(bm25, dense, weight_bm25=0.5).search("q", top_k=3)
    assert hits[0].chunk.id == "B"
    assert all(0.0 <= h.score <= 1.0 for h in hits)


def test_hybrid_rejects_a_weight_outside_zero_one():
    with pytest.raises(ValueError):
        _stub_hybrid(1.5).search("q")


def test_three_retrievers_find_the_matching_chunk():
    pytest.importorskip("rank_bm25")
    retrievers = build_retrievers(_chunks(), FakeEmbedder(), persist_dir=None)
    assert set(retrievers) == {"bm25", "dense", "hybrid"}
    for weight in (0.0, 0.5, 1.0):
        retrievers["hybrid"].weight_bm25 = weight
        for name, retriever in retrievers.items():
            hits = retriever.search("conservar insulina nevera", top_k=3)
            assert hits[0].chunk.id == "c1", (name, weight)
    assert BM25Retriever([]).search("insulina") == []
