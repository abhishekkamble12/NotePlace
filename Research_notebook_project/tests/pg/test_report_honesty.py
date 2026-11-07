"""The rendered report must describe what actually happened (real Postgres, real render_node + reporter)."""
import pytest

from app.models import Claim, Job, Source, SubQuestion
from app.research import nodes

pytestmark = pytest.mark.int


@pytest.mark.asyncio
async def test_report_distinguishes_dropped_from_never_drafted_and_shows_final_status(sm_factory, monkeypatch):
    sm = sm_factory()
    async with sm() as s:
        job = Job(question="q", status="running", stage="analyzing")
        s.add(job)
        await s.flush()
        kept_sq = SubQuestion(job_id=job.id, text="has a kept claim", status="done")
        dropped_sq = SubQuestion(job_id=job.id, text="only dropped claims", status="done")
        empty_sq = SubQuestion(job_id=job.id, text="nothing drafted", status="done")
        src = Source(job_id=job.id, url="http://x", domain="x", content_hash="h", snapshot="s")
        s.add_all([kept_sq, dropped_sq, empty_sq, src])
        await s.flush()
        s.add_all([
            Claim(job_id=job.id, subq_id=kept_sq.id, source_id=src.id, text="kept one", quote="q", final_status="kept"),
            Claim(job_id=job.id, subq_id=dropped_sq.id, source_id=src.id, text="dropped one", quote="q", final_status="dropped"),
        ])
        await s.commit()
        jid = job.id

    monkeypatch.setattr(nodes, "AsyncSessionLocal", sm)
    out = await nodes.render_node({"job_id": jid})
    async with sm() as s:
        report = (await s.get(Job, jid)).report_md

    assert out["status"] == "done"
    assert "**Status:** `done`" in report and "running" not in report.split("\n")[2]
    assert "Claims were drafted but none passed verification" in report
    assert "No claims were drafted" in report
    assert "**No claims drafted:** *'nothing drafted'*" in report
    assert "**No claims drafted:** *'only dropped claims'*" not in report
    assert "Every sub-question was researched" not in report
