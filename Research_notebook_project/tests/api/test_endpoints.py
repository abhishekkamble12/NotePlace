"""API tests: real FastAPI app over ASGI, real Postgres. Only the background pipeline launcher is replaced
(recorded; listed in the TESTING.md mock table)."""
import csv
import io
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import select

import app.main as main
from app.db import get_db
from app.models import Claim, Job, Source

pytestmark = pytest.mark.int


@pytest.fixture
def client(sm_factory, monkeypatch):
    sm = sm_factory()
    launched = AsyncMock()
    monkeypatch.setattr(main, "execute_research_job", launched)

    async def override():
        async with sm() as s:
            yield s
            await s.commit()

    main.app.dependency_overrides[get_db] = override
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t")
    c.sm, c.launched = sm, launched
    yield c
    main.app.dependency_overrides.clear()


async def _job(c, status="running", question="what is the efficiency of perovskite cells?"):
    async with c.sm() as s:
        j = Job(question=question, status=status)
        s.add(j)
        await s.commit()
        return j.id


@pytest.mark.asyncio
async def test_health_hits_the_real_db(client):
    r = await client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "db": True}


@pytest.mark.asyncio
async def test_post_research_creates_queued_job_and_launches_pipeline_once(client):
    r = await client.post("/research", json={"question": "does coffee cause cancer?", "max_tokens": 12345, "mode": "shadow"})
    assert r.status_code == 202
    jid = r.json()["job_id"]
    async with client.sm() as s:
        j = await s.get(Job, jid)
    assert (j.question, j.max_tokens, j.mode) == ("does coffee cause cancer?", 12345, "shadow")
    client.launched.assert_awaited_once_with(jid)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [
    {}, {"question": "abc"}, {"question": "valid question here", "mode": "nope"},
    {"question": "valid question here", "max_tokens": "lots"},
])
async def test_post_research_rejects_invalid_input_with_422_and_no_job(client, body):
    r = await client.post("/research", json=body)
    assert r.status_code == 422 and "detail" in r.json()
    async with client.sm() as s:
        assert (await s.execute(select(Job))).first() is None
    client.launched.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="no bounds on max_tokens/max_minutes: a caller can set an unbounded budget (Phase 3 ingress validation)")
@pytest.mark.parametrize("field,value", [("max_tokens", -1), ("max_tokens", 10**12), ("max_minutes", 0), ("max_minutes", 10**6)])
async def test_post_research_rejects_absurd_budgets(client, field, value):
    r = await client.post("/research", json={"question": "valid question here", field: value})
    assert r.status_code == 422


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="no question length cap (Phase 3 ingress)")
async def test_post_research_rejects_a_megabyte_question(client):
    r = await client.post("/research", json={"question": "a" * 1_000_000})
    assert r.status_code == 422


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/jobs/999999", "/jobs/999999/report"])
async def test_missing_job_is_a_clean_404(client, path):
    r = await client.get(path)
    assert r.status_code == 404 and r.json() == {"detail": "Job 999999 not found"}
    assert "Traceback" not in r.text


@pytest.mark.asyncio
async def test_job_status_counts_claims_from_the_database(client):
    jid = await _job(client)
    async with client.sm() as s:
        src = Source(job_id=jid, url="http://x", domain="x", content_hash="h", snapshot="s")
        s.add(src)
        await s.flush()
        for st in ("kept", "kept", "dropped"):
            s.add(Claim(job_id=jid, text="t", source_id=src.id, quote="q", final_status=st))
        await s.commit()
    j = (await client.get(f"/jobs/{jid}")).json()
    assert (j["total_claims"], j["verified_claims"], j["status"], j["tokens_spent"]) == (3, 2, "running", 0)


@pytest.mark.asyncio
async def test_cancel_sets_status(client):
    jid = await _job(client)
    assert (await client.post(f"/jobs/{jid}/cancel")).status_code == 200
    async with client.sm() as s:
        assert (await s.get(Job, jid)).status == "cancelled"


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="cancel unconditionally overwrites status, so a finished job becomes 'cancelled'")
async def test_cancel_does_not_rewrite_a_finished_job(client):
    jid = await _job(client, status="done")
    await client.post(f"/jobs/{jid}/cancel")
    async with client.sm() as s:
        assert (await s.get(Job, jid)).status == "done"


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s7 #9: no auth, /export/claims.csv dumps every job's question to anyone")
async def test_export_requires_credentials(client):
    await _job(client, question="confidential merger question")
    r = await client.get("/export/claims.csv")
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s7 #9: no auth, anyone can overwrite human labels")
async def test_labels_import_requires_credentials(client):
    r = await client.post("/labels", files={"file": ("l.csv", io.BytesIO(b"claim_id,human_label\n1,supported\n"), "text/csv")})
    assert r.status_code in (401, 403)


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="CSV formula injection: user-controlled question/claim text is exported unescaped")
async def test_export_csv_neutralizes_spreadsheet_formulas(client):
    jid = await _job(client, question='=HYPERLINK("http://evil","x")')
    async with client.sm() as s:
        s.add(Claim(job_id=jid, text="+cmd|calc!A0", quote="q", final_status="kept"))
        await s.commit()
    rows = list(csv.reader(io.StringIO((await client.get("/export/claims.csv")).text)))[1:]
    assert rows and all(not cell.startswith(("=", "+", "-", "@")) for row in rows for cell in row[2:4])


@pytest.mark.asyncio
async def test_labels_import_updates_only_existing_claims_and_ignores_blanks(client):
    jid = await _job(client)
    async with client.sm() as s:
        c1, c2 = Claim(job_id=jid, text="a", quote="q"), Claim(job_id=jid, text="b", quote="q")
        s.add_all([c1, c2])
        await s.commit()
        i1, i2 = c1.id, c2.id
    body = f"claim_id,human_label\n{i1},supported\n{i2},\n999999,unsupported\n".encode()
    r = await client.post("/labels", files={"file": ("l.csv", io.BytesIO(body), "text/csv")})
    assert r.json() == {"status": "ok", "updated_claims": 1}
    async with client.sm() as s:
        assert (await s.get(Claim, i1)).human_label == "supported" and not (await s.get(Claim, i2)).human_label


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="malformed labels CSV raises an unhandled error instead of a 4xx")
async def test_labels_import_malformed_csv_is_a_4xx(client):
    r = await client.post("/labels", files={"file": ("l.csv", io.BytesIO(b"claim_id,human_label\nNOTANUMBER,x\n"), "text/csv")})
    assert 400 <= r.status_code < 500


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="AUDIT s5: failed jobs store str(exception) + full traceback in report_md, served by GET /jobs/{id}/report")
async def test_failed_job_report_does_not_leak_exception_text_or_traceback(client, monkeypatch):
    import app.pipeline as pipeline
    jid = await _job(client, status="queued")
    monkeypatch.setattr(pipeline, "AsyncSessionLocal", client.sm)
    secret = "SECRET-PROMPT-TEXT-api_key=sk-12345"
    monkeypatch.setattr(pipeline.research_graph, "ainvoke", AsyncMock(side_effect=RuntimeError(secret)))
    await pipeline.execute_research_job(jid)
    r = await client.get(f"/jobs/{jid}/report")
    assert secret not in r.text and "Traceback" not in r.text
