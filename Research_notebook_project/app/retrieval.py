"""Notebook retrieval: dense (pgvector), sparse (PostgreSQL full-text), Reciprocal Rank Fusion, cross-encoder rerank.

Tenant isolation: every query requires user_id AND notebook_id and filters in SQL (the notebook must belong to the
user) BEFORE ranking. There is no vector index on purpose: an ANN index picks its top-k first and filters after,
which can silently return fewer or none of a tenant's rows. Exact scan is the correct trade-off at notebook scale.

Sparse retrieval is PostgreSQL full-text search (ts_rank_cd over a tsvector + GIN index), NOT BM25. Query terms are
OR-ed: websearch/plainto AND semantics return nothing for natural-language questions.

Dense queries use BGE's recommended retrieval instruction prefix, so the E1 baseline is a properly configured dense
retriever and not a weakened strawman.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import embed_text
from app.models import Chunk, Notebook

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MODES = ("dense", "hybrid", "hybrid+rerank")
CANDIDATES = 20
RRF_K = 60
_TERM_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class Hit:
    chunk_id: int
    source_id: int
    start_char: int
    end_char: int
    text: str
    page_start: int | None
    page_end: int | None
    section: str | None
    score: float


@dataclass
class Retrieval:
    hits: list[Hit]
    timings_ms: dict[str, float] = field(default_factory=dict)


def _scoped(user_id: int, notebook_id: int, chunker: str):
    """SELECT of Chunk rows restricted to this user's notebook and one chunker. The only place the filter lives."""
    if user_id is None or notebook_id is None:
        raise ValueError("user_id and notebook_id are required")
    return (select(Chunk).join(Notebook, Notebook.id == Chunk.notebook_id)
            .where(Notebook.id == notebook_id, Notebook.user_id == user_id, Chunk.chunker == chunker))


async def dense(db: AsyncSession, user_id: int, notebook_id: int, query_vec: list[float], chunker: str,
                k: int = CANDIDATES) -> list[Chunk]:
    stmt = (_scoped(user_id, notebook_id, chunker).where(Chunk.embedding.is_not(None))
            .order_by(Chunk.embedding.cosine_distance(query_vec), Chunk.id).limit(k))
    return list((await db.execute(stmt)).scalars())


def tsquery_text(query: str, max_terms: int = 32) -> str:
    """'a | b | c' from alphanumeric terms only (no tsquery operators can be injected through the user's text)."""
    seen: list[str] = []
    for t in _TERM_RE.findall(query.lower()):
        if t not in seen:
            seen.append(t)
    return " | ".join(seen[:max_terms])


async def sparse(db: AsyncSession, user_id: int, notebook_id: int, query: str, chunker: str,
                 k: int = CANDIDATES) -> list[Chunk]:
    q = tsquery_text(query)
    if not q:
        return []
    tsq = func.to_tsquery("english", q)
    stmt = (_scoped(user_id, notebook_id, chunker).where(Chunk.tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(Chunk.tsv, tsq).desc(), Chunk.id).limit(k))
    return list((await db.execute(stmt)).scalars())


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1. Ties break by id."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
    if not chunks:
        return []
    scores = _get_reranker().predict([(query, c.text) for c in chunks])
    return sorted(zip(chunks, (float(s) for s in scores), strict=True), key=lambda cs: (-cs[1], cs[0].id))


def _hit(c: Chunk, score: float) -> Hit:
    return Hit(c.id, c.source_id, c.start_char, c.end_char, c.text, c.page_start, c.page_end, c.section, score)


async def retrieve(db: AsyncSession, user_id: int, notebook_id: int, query: str, mode: str = "hybrid+rerank",
                   chunker: str = "structure", k: int = 10) -> Retrieval:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    qv = embed_text(BGE_QUERY_PREFIX + query)
    t["embed"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    d = await dense(db, user_id, notebook_id, qv, chunker)
    t["dense"] = (time.perf_counter() - t0) * 1000
    if mode == "dense":
        return Retrieval([_hit(c, float(len(d) - i)) for i, c in enumerate(d[:k])], t)  # score = rank-based, not a similarity
    t0 = time.perf_counter()
    s = await sparse(db, user_id, notebook_id, query, chunker)
    t["sparse"] = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    by_id = {c.id: c for c in [*d, *s]}
    fused = rrf([[c.id for c in d], [c.id for c in s]])[:CANDIDATES]
    t["fuse"] = (time.perf_counter() - t0) * 1000
    if mode == "hybrid":
        return Retrieval([_hit(by_id[i], sc) for i, sc in fused[:k]], t)
    t0 = time.perf_counter()
    ranked = rerank(query, [by_id[i] for i, _ in fused])
    t["rerank"] = (time.perf_counter() - t0) * 1000
    return Retrieval([_hit(c, sc) for c, sc in ranked[:k]], t)
