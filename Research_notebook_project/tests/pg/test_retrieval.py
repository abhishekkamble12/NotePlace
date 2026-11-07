"""write_node retrieval on real Postgres + pgvector: chunks are ranked by cosine distance to the sub-question."""
from unittest.mock import AsyncMock

import pytest

from app.models import Chunk, Job, Source, SubQuestion
from app.research import nodes

pytestmark = pytest.mark.int


def unit(i):
    v = [0.0] * 384
    v[i] = 1.0
    return v


@pytest.mark.asyncio
async def test_write_node_feeds_the_most_similar_chunks_first(sm_factory, monkeypatch):
    sm = sm_factory()
    async with sm() as s:
        job = Job(question="q", status="running")
        s.add(job)
        await s.flush()
        sq = SubQuestion(job_id=job.id, text="sub-question", status="done")
        src = Source(job_id=job.id, url="http://x", domain="x", content_hash="h", snapshot="s" * 100)
        s.add_all([sq, src])
        await s.flush()
        far = Chunk(source_id=src.id, subq_id=sq.id, text="far", start_char=0, end_char=3, embedding=unit(0))
        best = Chunk(source_id=src.id, subq_id=sq.id, text="best", start_char=10, end_char=14, embedding=unit(1))
        mid = Chunk(source_id=src.id, subq_id=sq.id, text="mid", start_char=20, end_char=23,
                    embedding=[0.0, 0.6, 0.8] + [0.0] * 381)
        no_vec = Chunk(source_id=src.id, subq_id=sq.id, text="no vector", start_char=30, end_char=39, embedding=None)
        s.add_all([far, best, mid, no_vec])
        await s.commit()
        ids = {"far": far.id, "best": best.id, "mid": mid.id, "no_vec": no_vec.id}
        state = {"job_id": job.id, "subquestions": [{"id": sq.id, "text": "sub-question", "status": "done"}], "claims": []}

    monkeypatch.setattr(nodes, "AsyncSessionLocal", sm)
    monkeypatch.setattr(nodes, "embed_text", lambda text: unit(1))
    draft = AsyncMock(return_value=[])
    monkeypatch.setattr(nodes, "draft_claims_for_subquestion", draft)

    await nodes.write_node(state)

    got = [e["chunk_id"] for e in draft.call_args.kwargs["evidence_snippets"]]
    assert got == [ids["best"], ids["mid"], ids["far"]]  # cosine order; the chunk without a vector is excluded
