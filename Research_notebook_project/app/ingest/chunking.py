"""Chunkers. Every chunk is (start, end, section, page_start, page_end) with text == snapshot[start:end].

- fixed_chunks: the E1 baseline, the existing word-window chunker (350 words, overlap 50), no structure.
- structure_chunks: whole paragraphs packed up to MAX_TOKENS, never ending on a heading, overlap = trailing whole
  blocks of at most OVERLAP_MAX tokens. Pages come from page_of(), sections from the nearest preceding heading.

MAX_TOKENS is 480, not the 600 in the project doc: bge-small truncates input at 512 tokens, so a longer chunk would
be embedded from its first 512 tokens only while the text shown to the writer is longer.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.core.fetcher import chunk_spans
from app.ingest.pdf import Block, ParsedPdf, page_of, split_long_block

MIN_TOKENS = 300
MAX_TOKENS = 480
OVERLAP_MAX = 100
HARD_MAX_TOKENS = 512   # bge-small input limit


@dataclass(frozen=True)
class ChunkSpan:
    start: int
    end: int
    section: str | None
    page_start: int | None
    page_end: int | None


def bge_token_counter() -> Callable[[str], int]:
    """Token counter of the actual embedding model, so chunk sizes mean what the doc says they mean."""
    from app.core.embeddings import get_embedding_model
    tok = get_embedding_model().tokenizer
    return lambda t: len(tok.encode(t, add_special_tokens=False))


def _span(start: int, end: int, section: str | None, offsets: list[int] | None) -> ChunkSpan:
    ps = page_of(start, offsets) if offsets else None
    pe = page_of(end - 1, offsets) if offsets else None
    return ChunkSpan(start, end, section, ps, pe)


def fixed_chunks(snapshot: str, page_offsets: list[int] | None = None) -> list[ChunkSpan]:
    return [_span(a, b, None, page_offsets) for a, b in chunk_spans(snapshot, chunk_size=350, overlap=50)]


def structure_chunks(doc: ParsedPdf, count: Callable[[str], int] | None = None) -> list[ChunkSpan]:
    count = count or bge_token_counter()
    snap = doc.snapshot
    # Units: headings stay whole; over-long paragraphs are split into sentence-group spans.
    units: list[Block] = []
    for b in doc.blocks:
        if b.kind == "para" and count(snap[b.start:b.end]) > MAX_TOKENS:
            units += [Block(s, e, "para") for s, e in split_long_block(snap[b.start:b.end], b.start, MAX_TOKENS, count)]
        else:
            units.append(b)
    toks = [count(snap[u.start:u.end]) for u in units]

    out: list[ChunkSpan] = []
    i, n = 0, len(units)
    section: str | None = None
    while i < n:
        j, total, sec_at_start = i, 0, section
        while j < n and total + toks[j] <= MAX_TOKENS:
            total += toks[j]
            j += 1
        if j == i:  # a unit alone exceeds the limit only if the splitter failed; take it rather than loop forever
            j = i + 1
        if j == i + 1 and units[i].kind == "heading" and j < n and toks[i] + toks[j] <= HARD_MAX_TOKENS:
            j += 1  # a heading is never a chunk by itself: let it overflow MAX_TOKENS up to the embedding limit
        while j - i > 1 and units[j - 1].kind == "heading":  # never strand a heading at the end of a chunk
            j -= 1
        out.append(_span(units[i].start, units[j - 1].end, _last_heading(snap, units, i, sec_at_start), doc.page_offsets))
        for u in units[i:j]:
            if u.kind == "heading":
                section = snap[u.start:u.end]
        if j >= n:
            break
        k, ov = j, 0  # overlap: step back over whole trailing blocks of the chunk just emitted, never to its first unit
        # the overlap must leave room for the next unit, else the next chunk would be a redundant subset of this one
        while k - 1 > i and ov + toks[k - 1] <= OVERLAP_MAX and ov + toks[k - 1] + toks[j] <= MAX_TOKENS and units[k - 1].kind != "heading":
            k -= 1
            ov += toks[k]
        i = k if k > i else j
    return out


def _last_heading(snap: str, units: list[Block], i: int, carried: str | None) -> str | None:
    """Section label of a chunk: its first heading if it starts with one, else the section carried in from before."""
    u = units[i]
    return snap[u.start:u.end] if u.kind == "heading" else carried
