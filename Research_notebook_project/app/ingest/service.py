"""Write path: PDF bytes -> immutable Source snapshot + embedded chunks, inside one notebook.

Idempotent: the same document (same snapshot hash) in the same notebook is stored once. Chunks are added per chunker
("fixed" = E1 baseline, "structure" = production), so an experiment can index the same source both ways.
"""
from __future__ import annotations

import hashlib

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import embed_texts
from app.ingest.chunking import ChunkSpan, bge_token_counter, fixed_chunks, structure_chunks
from app.ingest.pdf import PdfError, parse_pdf
from app.models import Chunk, Notebook, Source, User

CHUNKERS = ("fixed", "structure")


async def get_or_create_user(db: AsyncSession, external_sub: str) -> User:
    u = (await db.execute(select(User).where(User.external_sub == external_sub))).scalar_one_or_none()
    if u is None:
        u = User(external_sub=external_sub)
        db.add(u)
        await db.flush()
    return u


async def create_notebook(db: AsyncSession, user_id: int, title: str = "") -> Notebook:
    nb = Notebook(user_id=user_id, title=title)
    db.add(nb)
    await db.flush()
    return nb


async def ingest_pdf(db: AsyncSession, user_id: int, notebook_id: int, data: bytes, filename: str = "",
                     chunkers: tuple[str, ...] = ("structure",)) -> tuple[Source, bool]:
    """Returns (source, created). Raises PdfError for unreadable/scanned input and LookupError if the notebook is not
    the user's (the ownership check is in SQL, not in the caller)."""
    bad = set(chunkers) - set(CHUNKERS)
    if bad:
        raise ValueError(f"unknown chunker(s): {sorted(bad)}")
    owned = (await db.execute(select(Notebook.id).where(Notebook.id == notebook_id, Notebook.user_id == user_id))).first()
    if owned is None:
        raise LookupError("notebook not found")

    doc = parse_pdf(data)
    h = hashlib.sha256(doc.snapshot.encode("utf-8")).hexdigest()
    src = (await db.execute(select(Source).where(Source.notebook_id == notebook_id, Source.content_hash == h))).scalar_one_or_none()
    created = src is None
    if created:
        src = Source(job_id=None, notebook_id=notebook_id, kind="pdf", title=doc.title or filename, url=f"pdf:{filename}"[:1000],
                     domain="", content_hash=h, snapshot=doc.snapshot, page_offsets=doc.page_offsets)
        db.add(src)
        await db.flush()

    have = set((await db.execute(select(Chunk.chunker).where(Chunk.source_id == src.id).distinct())).scalars())
    for name in chunkers:
        if name in have:
            continue
        spans: list[ChunkSpan] = (fixed_chunks(src.snapshot, src.page_offsets) if name == "fixed"
                                  else structure_chunks(doc, bge_token_counter()))
        texts = [src.snapshot[s.start:s.end] for s in spans]
        for s, t, emb in zip(spans, texts, embed_texts(texts), strict=True):
            db.add(Chunk(source_id=src.id, notebook_id=notebook_id, text=t, start_char=s.start, end_char=s.end, embedding=emb,
                         section=s.section, page_start=s.page_start, page_end=s.page_end, chunker=name))
    await db.flush()
    return src, created


__all__ = ["PdfError", "create_notebook", "get_or_create_user", "ingest_pdf"]
