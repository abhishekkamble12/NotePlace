"""One test per OPEN AUDIT finding (AUDIT.md at 569fc72, ids numbered in TESTING.md). Each reproduces the bug today and is
xfail(strict=True): the build FAILS when the bug is fixed, which forces removing the xfail and keeping the test as a regression.
Fixed findings keep their regression tests where they were written (see the finding map in TESTING.md).
Network boundaries (search, fetch, embeddings) are replaced with local stubs; the nodes, the DB and the graph code are real."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import func, select, update

from app.core.llm import SlidingWindowRateLimiter
from app.core.verifier import EntailmentVerdict
from app.models import Chunk, Claim, Job, Source, SubQuestion
from app.research import nodes
from app.research.graph import build_research_graph

pytestmark = pytest.mark.int

VEC = [0.1] * 384
PAGE = ("Perovskite cells reached 33.9% efficiency in a late 2023 lab test. " * 100).strip()


async def _seed_job(sm, n_subqs=3):
    async with sm() as s:
        job = Job(question="q", status="running", max_tokens=100_000)
        s.add(job)
        await s.flush()
        sqs = [SubQuestion(job_id=job.id, text=f"sub question {i}", status="pending") for i in range(n_subqs)]
        s.add_all(sqs)
        await s.commit()
        return job.id, [{"id": q.id, "text": q.text, "status": "pending"} for q in sqs]


def _wire(monkeypatch, sm, search, fetch):
    monkeypatch.setattr(nodes, "AsyncSessionLocal", sm)
    monkeypatch.setattr(nodes, "search_web", search)
    monkeypatch.setattr(nodes, "fetch_url_snapshot", fetch)
    monkeypatch.setattr(nodes, "embed_texts", lambda texts: [VEC for _ in texts])


async def _fetch_ok(url):
    return "example.org", "hash-" + url, PAGE


# ---------------------------------------------------------------- AUD-5c: cancellation is not honored inside a node
@pytest.mark.xfail(strict=True, reason="AUD-5c OPEN: cancel is only checked on the plan/research/write EDGES, never inside research_node's loop")
@pytest.mark.asyncio
async def test_AUD_5c_cancel_mid_research_stops_further_searches(sm_factory, monkeypatch):
    sm = sm_factory()
    job_id, subqs = await _seed_job(sm, 3)
    calls = []

    async def search(text, top_n=2):
        calls.append(text)
        if len(calls) == 1:  # the user cancels while the first sub-question is being researched
            async with sm() as s:
                await s.execute(update(Job).where(Job.id == job_id).values(status="cancelled"))
                await s.commit()
        return [f"http://example.org/{len(calls)}"]

    _wire(monkeypatch, sm, search, _fetch_ok)
    await nodes.research_node({"job_id": job_id, "subquestions": subqs, "max_tokens": 100_000, "source_ids": []})
    assert len(calls) == 1, f"research kept running after cancel: {len(calls)} searches"


# ---------------------------------------------------------------- AUD-8b: re-running a node duplicates chunks
@pytest.mark.xfail(strict=True, reason="AUD-8b OPEN: Source is deduped by content hash but Chunk inserts are not idempotent on node re-run")
@pytest.mark.asyncio
async def test_AUD_8b_rerunning_research_after_a_crash_does_not_duplicate_chunks(sm_factory, monkeypatch):
    sm = sm_factory()
    job_id, subqs = await _seed_job(sm, 1)

    async def search(text, top_n=2):
        return ["http://example.org/a"]

    _wire(monkeypatch, sm, search, _fetch_ok)
    state = {"job_id": job_id, "subquestions": subqs, "max_tokens": 100_000, "source_ids": []}
    await nodes.research_node(state)
    async with sm() as s:
        first = (await s.execute(select(func.count()).select_from(Chunk))).scalar_one()
        # crash after chunks were committed but before the sub-question was marked done
        await s.execute(update(SubQuestion).values(status="pending"))
        await s.commit()
    assert first > 1  # precondition: the page really produced several chunks
    await nodes.research_node(state)
    async with sm() as s:
        sources = (await s.execute(select(func.count()).select_from(Source))).scalar_one()
        second = (await s.execute(select(func.count()).select_from(Chunk))).scalar_one()
    assert sources == 1, "source dedupe by content hash must hold"
    assert second == first, f"chunks duplicated on re-run: {first} -> {second}"


# ---------------------------------------------------------------- AUD-9: exception text printed to stdout
@pytest.mark.xfail(strict=True, reason="AUD-9 OPEN: research_node print()s the raw exception text of a failed fetch")
@pytest.mark.asyncio
async def test_AUD_9_failed_fetch_does_not_print_exception_text_or_url_secrets(sm_factory, monkeypatch, capsys):
    sm = sm_factory()
    job_id, subqs = await _seed_job(sm, 1)

    async def search(text, top_n=2):
        return ["http://example.org/page?token=SECRET-TOKEN-8431"]

    async def boom(url):
        raise RuntimeError("upstream said: Authorization: Bearer SECRET-TOKEN-8431")

    _wire(monkeypatch, sm, search, boom)
    await nodes.research_node({"job_id": job_id, "subquestions": subqs, "max_tokens": 100_000, "source_ids": []})
    out = capsys.readouterr()
    assert "SECRET-TOKEN-8431" not in out.out + out.err


# ---------------------------------------------------------------- AUD-5a: MemorySaver, state dies with the process
@pytest.mark.xfail(strict=True, reason="AUD-5a OPEN: graph uses the in-memory MemorySaver; a crash loses graph state")
def test_AUD_5a_default_checkpointer_survives_a_process_restart():
    from langgraph.checkpoint.memory import MemorySaver
    assert not isinstance(build_research_graph().checkpointer, MemorySaver)


# ---------------------------------------------------------------- AUD-5e: limiter is per-process; TPD never enforced
@pytest.mark.xfail(strict=True, reason="AUD-5e OPEN: tpd_limit is stored but acquire() never reads it")
@pytest.mark.asyncio
async def test_AUD_5e_daily_token_limit_is_enforced():
    now = [0.0]
    slept = []

    async def sleep(s):
        slept.append(s)
        now[0] += s

    lim = SlidingWindowRateLimiter(tpm_limit=10**9, tpd_limit=100, clock_fn=lambda: now[0], sleep_fn=sleep)
    await lim.acquire(60, "m")
    now[0] += 120  # past the 60s window, still inside the day
    try:
        await asyncio.wait_for(lim.acquire(60, "m"), timeout=1)  # 120 tokens today > 100 limit
    except Exception:
        return  # raising or waiting both count as "enforced"
    assert slept, "second acquire sailed through: 120 tokens in one day with tpd_limit=100"


@pytest.mark.xfail(strict=True, reason="AUD-5e OPEN: limiter state is a per-process dict; two workers each get the full budget (needs a shared backend)")
@pytest.mark.asyncio
async def test_AUD_5e_two_limiter_instances_share_the_same_window():
    now = [0.0]
    slept = []

    async def sleep(s):
        slept.append(s)
        now[0] += s

    a = SlidingWindowRateLimiter(tpm_limit=100, clock_fn=lambda: now[0], sleep_fn=sleep)
    b = SlidingWindowRateLimiter(tpm_limit=100, clock_fn=lambda: now[0], sleep_fn=sleep)  # stand-in for a second worker
    await a.acquire(80, "m")
    await b.acquire(80, "m")  # together 160 in the same minute against a 100 TPM provider limit
    assert slept, "second worker was admitted immediately: combined 160 tokens/min against a 100 limit"


# ---------------------------------------------------------------- AUD-6 (FIXED): real DB + real verify_claim routing
@pytest.mark.asyncio
async def test_AUD_6_regression_verdict_routing_on_real_postgres(sm_factory, monkeypatch):
    """supports -> kept, partial -> needs_rewrite (once), unsupported -> dropped and NEVER rewritten.
    Real Postgres rows and the real verify_claim; only the LLM call inside verify_claim is a local stub."""
    sm = sm_factory()
    snap = "Intro sentence. The Acme lab measured 33.9% efficiency in 2023 using a perovskite cell. Closing sentence."
    q = "The Acme lab measured 33.9% efficiency in 2023 using a perovskite cell."
    st = snap.index(q)
    async with sm() as s:
        job = Job(question="q", status="running")
        s.add(job)
        await s.flush()
        src = Source(job_id=job.id, url="http://example.org", domain="example.org", content_hash="h", snapshot=snap)
        s.add(src)
        await s.flush()
        ids = {}
        for name in ("supports", "partial", "unsupported"):
            c = Claim(job_id=job.id, source_id=src.id, text="Acme measured 33.9% in 2023.", quote=q, start_char=st,
                      end_char=st + len(q), final_status="pending", rewrite_count=0)
            s.add(c)
            await s.flush()
            ids[name] = c.id
        await s.commit()
        job_id = job.id

    verdicts = iter(["supports", "partial", "unsupported"])  # verify_node walks claims in id order
    llm_stub = AsyncMock(side_effect=lambda **kw: EntailmentVerdict(verdict=next(verdicts), reason="r"))
    monkeypatch.setattr(nodes, "AsyncSessionLocal", sm)
    with patch("app.core.verifier.call_llm", llm_stub):
        await nodes.verify_node({"job_id": job_id, "mode": "standard", "claims": []})
    async with sm() as s:
        got = {name: (await s.get(Claim, cid)).final_status for name, cid in ids.items()}
    assert got == {"supports": "kept", "partial": "needs_rewrite", "unsupported": "dropped"}
    assert llm_stub.await_count == 3
