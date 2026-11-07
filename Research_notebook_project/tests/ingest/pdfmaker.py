"""Builds small real PDFs in memory (PyMuPDF) with known text, headings and page boundaries."""
import pymupdf


def make_pdf(pages: list[list[tuple[str, float]]], title: str = "") -> bytes:
    """pages = [[(paragraph_text, font_size), ...], ...]; size 18 = heading, 10 = body."""
    doc = pymupdf.open()
    for page_blocks in pages:
        page = doc.new_page()
        y = 80.0
        for text, size in page_blocks:
            rect = pymupdf.Rect(60, y, 540, y + 400)
            used = page.insert_textbox(rect, text, fontsize=size, fontname="helv")
            assert used >= 0, "test PDF block overflowed"
            lines = max(1, int(len(text) * size * 0.5 / 480) + 1)
            y += lines * size * 1.25 + 14
    if title:
        doc.set_metadata({"title": title})
    return doc.tobytes()


def words(n: int, tag: str) -> str:
    return " ".join(f"{tag}{i}" for i in range(n)) + "."
