"""PDF parsing and chunking: offsets are exact slices of the snapshot; pages and sections are derived, not guessed."""
import pymupdf
import pytest

from app.ingest import chunking
from app.ingest.chunking import fixed_chunks, structure_chunks
from app.ingest.pdf import PdfError, page_of, parse_pdf, split_long_block
from tests.ingest.pdfmaker import make_pdf, words

wc = lambda t: len(t.split())  # noqa: E731  whitespace "tokens" keep these unit tests model-free

PAGES = [
    [("Intro Heading", 18), ("Alpha beta gamma first paragraph of page one.", 10)],
    [("Methods Heading", 18), ("Delta epsilon second page paragraph about methods.", 10), ("Third paragraph also on page two.", 10)],
    [("Results Heading", 18), ("Zeta eta theta final page paragraph with results.", 10)],
]


@pytest.fixture(scope="module")
def parsed():
    return parse_pdf(make_pdf(PAGES, title="Test Doc"))


def test_blocks_are_exact_snapshot_slices_and_ordered(parsed):
    prev_end = 0
    for b in parsed.blocks:
        assert 0 <= prev_end <= b.start < b.end <= len(parsed.snapshot)
        prev_end = b.end
    texts = [parsed.snapshot[b.start:b.end] for b in parsed.blocks]
    assert "Intro Heading" in texts and "Third paragraph also on page two." in texts


def test_headings_detected_by_font_size(parsed):
    heads = [parsed.snapshot[b.start:b.end] for b in parsed.blocks if b.kind == "heading"]
    assert heads == ["Intro Heading", "Methods Heading", "Results Heading"]


def test_page_numbers_derived_from_offsets(parsed):
    assert parsed.n_pages == 3 and len(parsed.page_offsets) == 3
    for needle, page in [("Alpha beta", 1), ("Delta epsilon", 2), ("Third paragraph", 2), ("Zeta eta", 3)]:
        assert page_of(parsed.snapshot.index(needle), parsed.page_offsets) == page


def test_metadata_title(parsed):
    assert parsed.title == "Test Doc"


def test_empty_page_never_attributed():
    doc = parse_pdf(make_pdf([[("A", 18), ("page one text is here and long enough", 10)], [], [("B", 18), ("page three text is here and long enough", 10)]]))
    assert page_of(doc.snapshot.index("page three"), doc.page_offsets) == 3
    assert page_of(doc.snapshot.index("page one"), doc.page_offsets) == 1


def test_scanned_pdf_without_text_layer_rejected():
    d = pymupdf.open()
    d.new_page()
    d.new_page()
    with pytest.raises(PdfError, match="no text layer"):
        parse_pdf(d.tobytes())


@pytest.mark.parametrize("data", [b"", b"not a pdf at all", b"%PDF-1.4 garbage"])
def test_garbage_rejected(data):
    with pytest.raises(PdfError):
        parse_pdf(data)


def test_structure_chunks_offsets_roundtrip_and_metadata(parsed):
    chunks = structure_chunks(parsed, wc)
    assert chunks
    for c in chunks:
        text = parsed.snapshot[c.start:c.end]
        assert text.strip() and text == text.strip()
        assert c.page_start == page_of(c.start, parsed.page_offsets)
        assert c.page_end == page_of(c.end - 1, parsed.page_offsets)
    assert chunks[0].section == "Intro Heading"
    # nothing is lost: every block is inside some chunk
    for b in parsed.blocks:
        assert any(c.start <= b.start and b.end <= c.end for c in chunks)


def _long_doc():
    return parse_pdf(make_pdf([[("Big Heading", 18), (words(60, "a"), 10), (words(60, "b"), 10)],
                               [(words(60, "c"), 10), ("Next Heading", 18), (words(60, "d"), 10)]]))


def test_chunks_respect_max_and_never_end_on_heading(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_TOKENS", 150)
    monkeypatch.setattr(chunking, "OVERLAP_MAX", 70)
    doc = _long_doc()
    chunks = structure_chunks(doc, wc)
    assert len(chunks) >= 2
    heading_ends = {b.end for b in doc.blocks if b.kind == "heading"}
    for c in chunks:
        assert wc(doc.snapshot[c.start:c.end]) <= 150
        assert c.end not in heading_ends
    # consecutive chunks overlap by whole blocks, within the cap, and always advance
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert b.start > a.start and b.end > a.end
        assert wc(doc.snapshot[b.start:a.end]) <= 70 if b.start < a.end else True


def test_section_label_follows_headings(monkeypatch):
    monkeypatch.setattr(chunking, "MAX_TOKENS", 100)
    doc = _long_doc()
    chunks = structure_chunks(doc, wc)
    assert chunks[0].section == "Big Heading"
    last = chunks[-1]
    assert "d0" in doc.snapshot[last.start:last.end]
    assert last.section in ("Next Heading", "Big Heading")  # section at chunk start; never invented


def test_overlong_paragraph_is_split_on_sentences_with_exact_spans():
    text = " ".join(f"Sentence number {i} has some words." for i in range(40))
    parts = split_long_block(text, 1000, 60, wc)
    assert len(parts) > 1
    for a, b in parts:
        piece = text[a - 1000:b - 1000]
        assert wc(piece) <= 60 and piece == piece.strip()
    assert parts[0][0] == 1000 and parts[-1][1] == 1000 + len(text)


def test_overlong_single_sentence_falls_back_to_whitespace_windows():
    text = " ".join(f"w{i}" for i in range(100))
    parts = split_long_block(text, 0, 30, wc)
    assert all(wc(text[a:b]) <= 30 for a, b in parts) and len(parts) == 4


def test_fixed_chunks_are_the_e1_baseline(parsed):
    chunks = fixed_chunks(parsed.snapshot, parsed.page_offsets)
    assert chunks[0].start == 0 and all(c.section is None for c in chunks)
    assert chunks[-1].end == len(parsed.snapshot.rstrip())


def test_real_bge_tokenizer_keeps_chunks_under_the_embedding_limit():
    doc = parse_pdf(make_pdf([[("H", 18)] + [(words(120, f"t{i}x"), 10) for i in range(3)]]))
    count = chunking.bge_token_counter()
    for c in structure_chunks(doc, count):
        assert count(doc.snapshot[c.start:c.end]) <= chunking.MAX_TOKENS <= 512
