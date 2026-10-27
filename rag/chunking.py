"""Data indexing, step 2: chunking.

Two strategies, both aware of the Markdown structure of the corpus (one file per medicine,
one ``##`` section per data-sheet section):

* ``recursive_chunks``: split by paragraphs, then lines, then sentences... until every chunk
  fits ``chunk_size``. It is the default of most RAG systems.
* ``semantic_chunks``: embed every sentence and start a new chunk where the meaning changes
  (cosine distance between consecutive sentences above a threshold).

Two options apply to both, so that the comparison is fair and so that each one can be
switched off for an ablation:

* ``by_section``: never let a chunk cross a Markdown heading. Without it a chunk can mix the
  end of one section ("6.3 Periodo de validez") with the start of the next one.
* ``add_headers``: prepend the heading of the section (``MEDICAMENTO — 4.2 Posología``) to
  every chunk. A chunk from the middle of a section does not say which medicine it talks
  about, and neither BM25 nor the embedding model can guess it; the header fixes that. The
  original text stays contiguous inside the chunk, so a verbatim passage copied from the
  corpus is still found by plain substring search.

Compare both strategies with ``rag/evaluate.py`` (number of chunks, average length and
Recall@k on your gold set). To choose the semantic threshold, run::

    uv run python -m rag.chunking --doc ceftriaxona --sweep 0.15 0.2 0.25 0.3 0.4

which prints the distribution of distances, the chunks each threshold produces, and saves
the plot of distances for one document (``reports/chunking_distances.png``).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.ingest import CORPUS_DIR, Document, load_corpus

EmbedFn = Callable[[list[str]], np.ndarray]

# Regular expressions: the last-but-one one is a lookbehind, so the punctuation stays with
# its sentence and only the whitespace after it is used as the cut.
RECURSIVE_SEPARATORS = ("\n\n", "\n", r"(?<=[.!?])\s+", " ", "")

_HEADING = re.compile(r"^#{1,6}[ \t]+(.+?)[ \t#]*$")
# A sentence ends at . ! ? followed by a capital letter (or an opening mark). Requiring the
# capital keeps "p. ej.", "máx. 5 mg" and "0,9 %" in one piece.
_SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+(?=[A-ZÁÉÍÓÚÜÑ¿¡\"“«(\[])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document
    heading: str = ""  # section the chunk comes from (already prepended to ``text`` if used)


# ------------------------------------------------------------------------------ structure


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split a Markdown text into ``(heading, body)`` pairs, in order.

    The text before the first heading is returned with an empty heading. Sections without
    body are dropped. Bodies are exact substrings of ``text`` (stripped).
    """
    heading = ""
    lines: list[str] = []
    raw: list[tuple[str, list[str]]] = []
    for line in text.split("\n"):
        match = _HEADING.match(line)
        if match:
            raw.append((heading, lines))
            heading, lines = match.group(1).strip(), []
        else:
            lines.append(line)
    raw.append((heading, lines))
    sections = []
    for head, body_lines in raw:
        body = "\n".join(body_lines).strip()
        if body:
            sections.append((head, body))
    return sections


def _sections(doc: Document, by_section: bool) -> list[tuple[str, str]]:
    if by_section:
        return split_sections(doc.text)
    body = doc.text.strip()
    return [("", body)] if body else []


def _with_header(heading: str, body: str, add_headers: bool) -> str:
    return f"{heading}\n{body}" if add_headers and heading else body


def _budget(size: int, heading: str, add_headers: bool) -> int:
    """Characters left for the body once the header is counted (never below half of size)."""
    if not (add_headers and heading):
        return size
    return max(size - len(heading) - 1, size // 2)


# ------------------------------------------------------------------------------ recursive


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = RECURSIVE_SEPARATORS,
    by_section: bool = True,
    add_headers: bool = True,
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``.

    ``separators`` are regular expressions tried in order. ``chunk_size`` counts the header
    too (when there is one, the body gets what is left).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunks: list[Chunk] = []
    for doc in documents:
        pieces: list[tuple[str, str]] = []
        for heading, body in _sections(doc, by_section):
            budget = _budget(chunk_size, heading, add_headers)
            splitter = RecursiveCharacterTextSplitter(
                separators=list(separators),
                chunk_size=budget,
                chunk_overlap=min(chunk_overlap, budget // 2),
                length_function=len,
                is_separator_regex=True,
            )
            pieces.extend((heading, piece) for piece in splitter.split_text(body))
        for i, (heading, piece) in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{doc.id}-{i:04d}",
                    doc_id=doc.id,
                    source=doc.source,
                    text=_with_header(heading, piece, add_headers),
                    index=i,
                    heading=heading,
                )
            )
    return chunks


# ------------------------------------------------------------------------------- sentences


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _hard_split(text: str, start: int, end: int, max_chars: int) -> list[tuple[int, int]]:
    """Cut a span longer than ``max_chars`` on spaces (a table row, a list without stops)."""
    spans = []
    while end - start > max_chars:
        cut = text.rfind(" ", start + 1, start + max_chars + 1)
        if cut <= start:
            cut = start + max_chars
        spans.append(_trim(text, start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    spans.append(_trim(text, start, end))
    return [s for s in spans if s[1] > s[0]]


def _sentence_spans(text: str, max_chars: int | None = None) -> list[tuple[int, int]]:
    """``(start, end)`` offsets of every sentence. Each line break is also a boundary."""
    raw: list[tuple[int, int]] = []
    for line in re.finditer(r"[^\n]+", text):
        position = line.start()
        for match in _SENTENCE_END.finditer(line.group()):
            raw.append((position, line.start() + match.start()))
            position = line.start() + match.end()
        raw.append((position, line.end()))
    spans: list[tuple[int, int]] = []
    for start, end in raw:
        start, end = _trim(text, start, end)
        if end <= start:
            continue
        if max_chars and end - start > max_chars:
            spans.extend(_hard_split(text, start, end, max_chars))
        else:
            spans.append((start, end))
    return spans


def split_sentences(text: str) -> list[str]:
    """Sentence splitter for Spanish data-sheet text; every line break is a boundary too.

    It only cuts after ``. ! ?`` when a capital letter follows, so abbreviations written in
    lower case ("p. ej.", "máx. 5 mg") and decimals ("0,9 %") do not split a sentence.
    """
    return [text[a:b] for a, b in _sentence_spans(text)]


# ----------------------------------------------------------------------- semantic chunking


def cached_embed(embed_fn: EmbedFn) -> EmbedFn:
    """Wrap ``embed_fn`` so every distinct text is embedded only once.

    Useful when sweeping the threshold: the sentences are the same, only the cuts change.
    """
    cache: dict[str, np.ndarray] = {}

    def wrapped(texts: list[str]) -> np.ndarray:
        texts = list(texts)
        missing = list(dict.fromkeys(t for t in texts if t not in cache))
        if missing:
            for text, vector in zip(missing, np.asarray(embed_fn(missing)), strict=True):
                cache[text] = vector
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([cache[t] for t in texts])

    wrapped.cache = cache  # type: ignore[attr-defined]
    return wrapped


def sentence_distances(sentences: list[str], embed_fn: EmbedFn) -> np.ndarray:
    """Cosine distance ``1 - e_i · e_{i+1}`` between consecutive sentences (length n - 1)."""
    if len(sentences) < 2:
        return np.zeros(0, dtype=np.float32)
    return _consecutive_distances(np.asarray(embed_fn(sentences), dtype=np.float32))


def _consecutive_distances(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.clip(1.0 - np.sum(embeddings[:-1] * embeddings[1:], axis=1), 0.0, 2.0)


@dataclass
class _Section:
    doc: Document
    heading: str
    body: str
    spans: list[tuple[int, int]]
    budget: int
    distances: np.ndarray | None = None


def _prepare(
    documents: list[Document], max_chars: int, by_section: bool, add_headers: bool
) -> list[_Section]:
    sections = []
    for doc in documents:
        for heading, body in _sections(doc, by_section):
            budget = _budget(max_chars, heading, add_headers)
            sections.append(_Section(doc, heading, body, _sentence_spans(body, budget), budget))
    return sections


def _embed_sections(sections: list[_Section], embed_fn: EmbedFn) -> None:
    """One embedding call for the whole corpus, then the distances of each section."""
    sentences = [s.body[a:b] for s in sections for a, b in s.spans]
    embeddings = np.asarray(embed_fn(sentences), dtype=np.float32) if sentences else None
    if embeddings is not None and len(embeddings) != len(sentences):
        raise ValueError("embed_fn must return one row per sentence")
    position = 0
    for section in sections:
        n = len(section.spans)
        if embeddings is None or n < 2:
            section.distances = np.zeros(0, dtype=np.float32)
        else:
            section.distances = _consecutive_distances(embeddings[position : position + n])
        position += n


def _group_spans(
    spans: list[tuple[int, int]],
    distances: np.ndarray,
    threshold: float,
    max_chars: int,
    min_chars: int,
) -> list[tuple[int, int]]:
    """Group consecutive sentences: close the group when the next one is too far or too long."""
    groups = []
    start, end = spans[0]
    for i in range(1, len(spans)):
        s, e = spans[i]
        if distances[i - 1] > threshold or e - start > max_chars:
            groups.append((start, end))
            start = s
        end = e
    groups.append((start, end))
    if min_chars > 0:
        merged: list[tuple[int, int]] = []
        for group in groups:
            if merged:
                prev_start, prev_end = merged[-1]
                too_small = min(prev_end - prev_start, group[1] - group[0]) < min_chars
                if too_small and group[1] - prev_start <= max_chars:
                    merged[-1] = (prev_start, group[1])
                    continue
            merged.append(group)
        groups = merged
    return groups


def semantic_chunks(
    documents: list[Document],
    embed_fn: EmbedFn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
    min_chunk_chars: int = 0,
    percentile: float | None = None,
    by_section: bool = True,
    add_headers: bool = True,
) -> list[Chunk]:
    """Group consecutive sentences while their embeddings stay close.

    ``embed_fn(list[str]) -> array of shape (n, d)`` with L2-normalised rows (see
    ``rag/embed.py``). The algorithm from the slides, applied inside each section:

    1. Split the section into sentences s_1 … s_n.
    2. Embed them: e_1 … e_n (one call for the whole corpus).
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars`` (header included).

    Extras, off by default so that the base algorithm stays the one from class:

    * ``min_chunk_chars``: merge a chunk shorter than this into its neighbour when the result
      still fits ``max_chunk_chars``.
    * ``percentile``: instead of a fixed threshold, use that percentile of all the distances
      of the corpus (e.g. 80 cuts at the 20 % largest jumps). It does not depend on the scale
      of the embedding model.

    Chunk text is copied verbatim from the document (line breaks included), so a passage
    copied from the corpus can be found inside a chunk by substring search.

    Plot the distances for one document with the threshold line (``plot_distances`` or
    ``python -m rag.chunking``) and include it in your report: it is the best way to justify
    the threshold you chose.
    """
    sections = _prepare(documents, max_chunk_chars, by_section, add_headers)
    _embed_sections(sections, embed_fn)

    threshold = distance_threshold
    if percentile is not None:
        parts = [s.distances for s in sections if s.distances is not None and len(s.distances)]
        if parts:
            threshold = float(np.percentile(np.concatenate(parts), percentile))

    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for section in sections:
        groups = _group_spans(
            section.spans, section.distances, threshold, section.budget, min_chunk_chars
        )
        for start, end in groups:
            i = counters.get(section.doc.id, 0)
            counters[section.doc.id] = i + 1
            chunks.append(
                Chunk(
                    id=f"{section.doc.id}-{i:04d}",
                    doc_id=section.doc.id,
                    source=section.doc.source,
                    text=_with_header(section.heading, section.body[start:end], add_headers),
                    index=i,
                    heading=section.heading,
                )
            )
    return chunks


def plot_distances(
    document: Document,
    embed_fn: EmbedFn,
    out_path: str | Path = "reports/chunking_distances.png",
    thresholds: tuple[float, ...] = (0.25,),
    max_chunk_chars: int = 1500,
) -> np.ndarray:
    """Plot the distance between consecutive sentences of one document, with the thresholds.

    Dashed grey lines mark where a new section of the document starts, so you can check
    whether the jumps in meaning match the structure. Returns the distances.
    """
    from matplotlib.figure import Figure

    sections = _prepare([document], max_chunk_chars, by_section=True, add_headers=False)
    sentences = [s.body[a:b] for s in sections for a, b in s.spans]
    distances = sentence_distances(sentences, embed_fn)

    figure = Figure(figsize=(11, 4))
    axes = figure.subplots()
    axes.plot(range(1, len(distances) + 1), distances, marker="o", markersize=3, linewidth=1)
    for threshold in thresholds:
        axes.axhline(threshold, linestyle="--", color="tab:red", label=f"umbral {threshold:g}")
    count = 0
    for k, section in enumerate(sections[:-1]):
        count += len(section.spans)
        label = "inicio de sección" if k == 0 else None
        axes.axvline(count, color="grey", alpha=0.4, linestyle=":", label=label)
    axes.set_xlabel("par de frases consecutivas (i, i+1)")
    axes.set_ylabel("distancia coseno")
    axes.set_title(f"Distancia entre frases consecutivas — {document.source}")
    axes.legend()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150, bbox_inches="tight")
    return distances


# ----------------------------------------------------------------------------------- stats


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table."""
    lengths = [len(c.text) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "median_chars": float(np.median(lengths)) if lengths else 0.0,
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
    }


def _format_stats(name: str, chunks: list[Chunk]) -> str:
    s = describe(chunks)
    return (
        f"{name:<28} {s['n_chunks']:>6} chunks | mean {s['mean_chars']:>6.0f} | "
        f"median {s['median_chars']:>6.0f} | min {s['min_chars']:>5} | max {s['max_chars']:>5}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunking statistics and plot of sentence distances to choose the threshold.",
    )
    parser.add_argument("--corpus", default=str(CORPUS_DIR))
    parser.add_argument("--doc", default=None, help="part of the file name to plot")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--sweep", type=float, nargs="*", default=[0.15, 0.2, 0.25, 0.3, 0.4])
    parser.add_argument("--max-chars", type=int, default=1500, help="semantic chunk size limit")
    parser.add_argument("--out", default="reports/chunking_distances.png")
    args = parser.parse_args()

    documents = load_corpus(Path(args.corpus))
    if not documents:
        print(f"No documents in {args.corpus}. Build the corpus first (rag/README.md).")
        return

    from rag.embed import Embedder

    embed_fn = cached_embed(Embedder().embed_passages)

    print(f"{len(documents)} documents\n")
    plain = recursive_chunks(documents, by_section=False, add_headers=False)
    print(_format_stats("recursive (plain)", plain))
    print(_format_stats("recursive (sections+headers)", recursive_chunks(documents)))

    sections = _prepare(documents, args.max_chars, True, True)
    _embed_sections(sections, embed_fn)
    all_distances = np.concatenate([s.distances for s in sections if len(s.distances)])
    quantiles = {q: float(np.percentile(all_distances, q)) for q in (10, 25, 50, 75, 90, 95)}
    shown = ", ".join(f"p{q}={v:.2f}" for q, v in quantiles.items())
    print(f"\nDistances between consecutive sentences (percentiles): {shown}")
    print(f"\nSemantic chunks (max {args.max_chars} chars):")
    for threshold in args.sweep:
        chunks = semantic_chunks(
            documents, embed_fn, distance_threshold=threshold, max_chunk_chars=args.max_chars
        )
        print(_format_stats(f"semantic, threshold {threshold:g}", chunks))

    document = next((d for d in documents if args.doc and args.doc in d.source), documents[0])
    plot_distances(
        document, embed_fn, args.out, thresholds=(args.threshold,), max_chunk_chars=args.max_chars
    )
    print(f"\nPlot of {document.source} saved to {args.out}")


if __name__ == "__main__":
    main()
