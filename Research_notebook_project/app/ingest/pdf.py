"""PDF -> immutable text snapshot with page boundaries and block structure (PyMuPDF, text layer only, no OCR).

The snapshot is the single source of truth: every block span satisfies snapshot[start:end] == block text, and
page_offsets[i] is the snapshot offset where page i+1 starts. Page numbers are DERIVED from offsets, never guessed.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean

import pymupdf

MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PAGES = 300
MARGIN_FRAC = 0.06       # blocks this close to the top/bottom edge and short are running headers/footers/page numbers
HEADING_SIZE_RATIO = 1.12
HEADING_MAX_CHARS = 150


class PdfError(ValueError):
    """Unreadable, too large, or no text layer. The message is safe to show to the caller."""


@dataclass(frozen=True)
class Block:
    start: int
    end: int
    kind: str   # "heading" | "para"


@dataclass(frozen=True)
class ParsedPdf:
    snapshot: str
    page_offsets: list[int]
    blocks: list[Block]
    title: str
    n_pages: int


def page_of(offset: int, page_offsets: list[int]) -> int:
    """1-based page containing snapshot[offset]. Empty pages share the next page's start, so they are never returned."""
    return max(1, sum(1 for p in page_offsets if p <= offset))


def _raw_blocks(page: pymupdf.Page) -> list[tuple[str, float, float, bool]]:
    """(text, mean font size, bold fraction, in_margin) per text block, in content-stream order."""
    h, out = page.rect.height, []
    for b in page.get_text("dict")["blocks"]:
        if b.get("type") != 0:
            continue
        lines, sizes, bold, chars = [], [], 0, 0
        for ln in b["lines"]:
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if not t:
                continue
            lines.append(t)
            for s in ln["spans"]:
                n = len(s["text"].strip())
                sizes += [s["size"]] * n
                chars += n
                bold += n if (s["flags"] & 16 or "bold" in s["font"].lower()) else 0
        if not lines:
            continue
        y0, y1 = b["bbox"][1], b["bbox"][3]
        in_margin = y1 < h * MARGIN_FRAC or y0 > h * (1 - MARGIN_FRAC)
        out.append((" ".join(lines), mean(sizes) if sizes else 0.0, bold / chars if chars else 0.0, in_margin))
    return out


def parse_pdf(data: bytes) -> ParsedPdf:
    if len(data) > MAX_PDF_BYTES:
        raise PdfError("PDF too large")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as e:  # pymupdf raises several types for corrupt input
        raise PdfError("not a readable PDF") from e
    if doc.needs_pass or doc.page_count == 0 or doc.page_count > MAX_PAGES:
        raise PdfError("encrypted, empty, or too many pages")

    pages = [_raw_blocks(p) for p in doc]
    pages = [[b for b in pg if not (b[3] and (len(b[0]) < 80 or b[0].strip().isdigit()))] for pg in pages]
    with_text = sum(1 for pg in pages if sum(len(b[0]) for b in pg) >= 20)
    if with_text < 0.5 * len(pages):
        raise PdfError("no text layer (scanned PDF); OCR is not supported")

    sizes = Counter()
    for pg in pages:
        for t, size, _, _ in pg:
            sizes[round(size, 1)] += len(t)
    body = sizes.most_common(1)[0][0]

    snapshot, offsets, blocks = "", [], []
    for pg in pages:
        offsets.append(len(snapshot) + (2 if snapshot and pg else 0))
        for t, size, bold, _ in pg:
            if snapshot:
                snapshot += "\n\n"
            is_head = len(t) <= HEADING_MAX_CHARS and not t.endswith((".", ",", ";")) and (
                size >= body * HEADING_SIZE_RATIO or (bold > 0.8 and size >= body * 0.98))
            blocks.append(Block(len(snapshot), len(snapshot) + len(t), "heading" if is_head else "para"))
            snapshot += t
    # An empty page records the offset where the next text begins; fix up any offset past the end
    offsets = [min(o, len(snapshot)) for o in offsets]

    meta_title = (doc.metadata or {}).get("title", "").strip()
    first_head = next((snapshot[b.start:b.end] for b in blocks if b.kind == "heading"), "")
    return ParsedPdf(snapshot, offsets, blocks, (meta_title or first_head)[:500], doc.page_count)


_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def split_long_block(text: str, start: int, max_tokens: int, count) -> list[tuple[int, int]]:
    """Split an over-long paragraph on sentence boundaries into (start, end) snapshot spans <= max_tokens each
    (a single sentence longer than the limit is split on whitespace). Spans are exact slices of the snapshot."""
    cuts, pos = [], 0
    for m in _SENT_RE.finditer(text):
        cuts.append((pos, m.start()))
        pos = m.end()
    cuts.append((pos, len(text)))
    out, cur = [], None
    for a, b in cuts:
        if count(text[a:b]) > max_tokens:  # pathological sentence: whitespace windows
            if cur:
                out.append(cur)
            words = [(m.start() + a, m.end() + a) for m in re.finditer(r"\S+", text[a:b])]
            i = 0
            while i < len(words):
                j = i + 1
                while j < len(words) and count(text[words[i][0]:words[j][1]]) <= max_tokens:
                    j += 1
                out.append((words[i][0], words[j - 1][1]))
                i = j
            cur = None
            continue
        if cur and count(text[cur[0]:b]) <= max_tokens:
            cur = (cur[0], b)
        else:
            if cur:
                out.append(cur)
            cur = (a, b)
    if cur:
        out.append(cur)
    return [(start + a, start + b) for a, b in out]
