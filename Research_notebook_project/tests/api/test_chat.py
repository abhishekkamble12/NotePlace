"""Notebook + grounded chat API: real FastAPI app over ASGI, real Postgres, real PDF parsing, embeddings and reranker.
Only the LLM provider is replaced (llm._call_provider), routed by system prompt."""
import json
import re

import httpx
import pytest
from sqlalchemy import select

import app.main as main
from app import chat
from app.core import llm
from app.core.verifier import VERIFIER_SYSTEM_PROMPT
from app.db import get_db
from app.models import LedgerEntry, Source
from tests.ingest.pdfmaker import make_pdf

pytestmark = pytest.mark.int

A, B = {"Authorization": "Bearer tok-a"}, {"Authorization": "Bearer tok-b"}
FACT = "The Zephyr reactor reached 42 percent efficiency in 2021."
PDF = make_pdf([
    [("Results", 18), (FACT + " It was assembled in Lyon by the Meridian group over three years.", 10)],
    [("Background", 18), ("Bird migration across northern Europe has been observed for several decades by volunteers.", 10)],
], title="Zephyr report")
INVENTED = "The Zephyr reactor reached 99 percent efficiency."


class FakeLLM:
    """Answers like the real models would, by prompt. verdict / abstain / bad_ref tweak behaviour per test."""

    def __init__(self, verdict="supports", abstain=False, bad_ref=False):
        self.verdict, self.abstain, self.bad_ref, self.calls = verdict, abstain, bad_ref, []

    async def __call__(self, spec, messages, max_tokens, response_format):
        system, user = messages[0]["content"], messages[1]["content"]
        if system == chat.CHAT_SYSTEM_PROMPT:
            self.calls.append("chat")
            if self.abstain:
                return json.dumps({"abstain": True, "sentences": []}), 100, 10
            # cite the evidence sentence that states FACT, by (chunk_id, sentence index), as the real model is told to
            m = re.search(r'chunk_id="(\d+)"(?:(?!</evidence_data>).)*?\n\[(\d+)\] ' + re.escape(FACT), user, re.S)
            cid, i = int(m.group(1)), int(m.group(2))
            sents = [{"claim": FACT, "chunk_id": cid + 10_000 if self.bad_ref else cid, "first_sentence": i, "last_sentence": i},
                     {"claim": INVENTED, "chunk_id": cid, "first_sentence": i, "last_sentence": i}]
            return json.dumps({"abstain": False, "sentences": sents}), 100, 20
        if system == chat.REWRITE_SYSTEM_PROMPT:
            self.calls.append("rewrite")
            return json.dumps({"question": "What efficiency did the Zephyr reactor reach?"}), 50, 10
        if system == VERIFIER_SYSTEM_PROMPT:
            self.calls.append("verify")
            return json.dumps({"verdict": self.verdict, "reason": "test"}), 80, 10
        raise AssertionError(f"unexpected prompt: {system[:60]}")


@pytest.fixture
def client(sm_factory, monkeypatch, tmp_path):
    sm = sm_factory()

    async def override():
        async with sm() as s:
            yield s
            await s.commit()

    monkeypatch.setenv("CITEGUARD_DEV_TOKENS", "tok-a:alice,tok-b:bob")
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)  # never read or write the real disk cache
    llm.rate_limiter.reset()
    main.app.dependency_overrides[get_db] = override
    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t", timeout=120)
    c.sm = sm
    yield c
    main.app.dependency_overrides.clear()


async def _setup(c, headers=A):
    nb = (await c.post("/notebooks", json={"title": "n"}, headers=headers)).json()["id"]
    r = await c.post(f"/notebooks/{nb}/sources", files={"file": ("z.pdf", PDF, "application/pdf")}, headers=headers)
    assert r.status_code == 201, r.text
    conv = (await c.post(f"/notebooks/{nb}/conversations", headers=headers)).json()["id"]
    return nb, conv


async def _ask(c, conv, fake, q="How efficient was the Zephyr reactor?", headers=A):
    llm._call_provider = fake  # conftest restores the real provider after each test
    r = await c.post(f"/conversations/{conv}/messages", json={"question": q}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_answer_keeps_verified_sentence_and_drops_invented_number(client):
    _, conv = await _setup(client)
    fake = FakeLLM()
    out = await _ask(client, conv, fake)

    assert not out["abstained"]
    [kept] = out["citations"]
    assert kept["text"] == FACT and kept["quote"] == FACT
    async with client.sm() as s:
        src = await s.get(Source, kept["source_id"])
    assert src.snapshot[kept["start_char"]:kept["end_char"]] == kept["quote"]  # quote copied from the snapshot by code
    assert kept["page"] == 1
    assert out["content"].endswith("[1]")
    assert [d["reason"] for d in out["dropped"]] == ["fact_token_mismatch"]  # 99 is not in the source span
    assert fake.calls == ["chat", "verify"]  # the invented sentence never reached Stage 2

    async with client.sm() as s:
        rows = (await s.execute(select(LedgerEntry).where(LedgerEntry.scope_type == "conversation",
                                                          LedgerEntry.scope_id == conv))).scalars().all()
    assert {r.phase for r in rows} == {"chat", "verifying"}


@pytest.mark.asyncio
async def test_unsupported_verdict_turns_answer_into_abstention(client):
    _, conv = await _setup(client)
    out = await _ask(client, conv, FakeLLM(verdict="unsupported"))
    assert out["abstained"] and out["abstain_reason"] == "no_verified_sentence"
    assert out["content"] == chat.ABSTAIN_TEXT and out["citations"] == []
    assert sorted(d["reason"] for d in out["dropped"]) == ["fact_token_mismatch", "unsupported"]


@pytest.mark.asyncio
async def test_partial_verdict_is_dropped_not_shown(client):
    _, conv = await _setup(client)
    out = await _ask(client, conv, FakeLLM(verdict="partial"))
    assert out["abstained"] and "partial" in [d["reason"] for d in out["dropped"]]


@pytest.mark.asyncio
async def test_model_abstention_is_recorded(client):
    _, conv = await _setup(client)
    out = await _ask(client, conv, FakeLLM(abstain=True))
    assert out["abstained"] and out["abstain_reason"] == "model_abstained"


@pytest.mark.asyncio
async def test_chunk_id_not_in_retrieved_evidence_is_rejected(client):
    _, conv = await _setup(client)
    fake = FakeLLM(bad_ref=True)
    out = await _ask(client, conv, fake)
    assert "invalid_citation_ref" in [d["reason"] for d in out["dropped"]]
    assert fake.calls == ["chat"]  # neither sentence reached Stage 2


@pytest.mark.asyncio
async def test_follow_up_is_rewritten_before_retrieval_and_history_is_listed(client):
    _, conv = await _setup(client)
    await _ask(client, conv, FakeLLM())
    fake = FakeLLM()
    out = await _ask(client, conv, fake, q="And where was it built?")
    assert fake.calls[0] == "rewrite"
    assert out["standalone_question"] == "What efficiency did the Zephyr reactor reach?"
    msgs = (await client.get(f"/conversations/{conv}/messages", headers=A)).json()
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]


@pytest.mark.asyncio
async def test_empty_notebook_abstains_without_calling_the_model(client):
    nb = (await client.post("/notebooks", json={}, headers=A)).json()["id"]
    conv = (await client.post(f"/notebooks/{nb}/conversations", headers=A)).json()["id"]
    fake = FakeLLM()
    out = await _ask(client, conv, fake)
    assert out["abstain_reason"] == "no_sources" and fake.calls == []


@pytest.mark.asyncio
async def test_retrieval_score_floor_abstains_before_the_model(client, monkeypatch):
    _, conv = await _setup(client)
    monkeypatch.setenv("CITEGUARD_ABSTAIN_MIN_RERANK_SCORE", "1000")
    fake = FakeLLM()
    out = await _ask(client, conv, fake)
    assert out["abstain_reason"] == "low_retrieval_score" and fake.calls == []


@pytest.mark.asyncio
async def test_user_daily_cap_spans_conversations_and_fails_closed(client, monkeypatch):
    nb, conv = await _setup(client)
    await _ask(client, conv, FakeLLM())
    async with client.sm() as s:
        used = sum(r.tokens_in + r.tokens_out for r in (await s.execute(select(LedgerEntry))).scalars())
    monkeypatch.setenv("CITEGUARD_USER_DAILY_TOKENS", str(used + 10))
    conv2 = (await client.post(f"/notebooks/{nb}/conversations", headers=A)).json()["id"]  # a new conversation does not reset it
    fake = FakeLLM()
    out = await _ask(client, conv2, fake, q="Who assembled the Zephyr reactor?")  # new prompt: not a free cache hit
    assert out["abstain_reason"] == "budget_exceeded" and fake.calls == []


@pytest.mark.asyncio
async def test_auth_required_and_other_users_resources_are_404(client):
    nb, conv = await _setup(client)
    assert (await client.get("/notebooks")).status_code == 401
    assert (await client.get("/notebooks", headers={"Authorization": "Bearer wrong"})).status_code == 401
    assert (await client.get("/notebooks", headers=B)).json() == []
    assert (await client.get(f"/notebooks/{nb}/sources", headers=B)).status_code == 404
    assert (await client.post(f"/notebooks/{nb}/conversations", headers=B)).status_code == 404
    assert (await client.post(f"/notebooks/{nb}/sources", files={"file": ("z.pdf", PDF)}, headers=B)).status_code == 404
    assert (await client.get(f"/conversations/{conv}/messages", headers=B)).status_code == 404
    r = await client.post(f"/conversations/{conv}/messages", json={"question": "steal it"}, headers=B)
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_no_dev_tokens_configured_means_everyone_is_401(client, monkeypatch):
    monkeypatch.delenv("CITEGUARD_DEV_TOKENS")
    assert (await client.get("/notebooks", headers=A)).status_code == 401


@pytest.mark.asyncio
async def test_other_users_notebook_is_never_retrieved(client):
    await _setup(client, headers=B)  # bob owns the only document
    nb = (await client.post("/notebooks", json={}, headers=A)).json()["id"]
    conv = (await client.post(f"/notebooks/{nb}/conversations", headers=A)).json()["id"]
    fake = FakeLLM()
    out = await _ask(client, conv, fake)
    assert out["abstain_reason"] == "no_sources" and fake.calls == []


@pytest.mark.asyncio
async def test_upload_rejects_non_pdf_oversize_and_unreadable(client, monkeypatch):
    nb = (await client.post("/notebooks", json={}, headers=A)).json()["id"]
    up = lambda data: client.post(f"/notebooks/{nb}/sources", files={"file": ("x.pdf", data, "application/pdf")}, headers=A)  # noqa: E731
    assert (await up(b"<html>not a pdf</html>")).status_code == 415
    assert (await up(b"%PDF-1.7 garbage")).status_code == 422
    monkeypatch.setattr(main, "MAX_PDF_BYTES", 100)
    assert (await up(PDF)).status_code == 413


@pytest.mark.asyncio
async def test_same_pdf_twice_is_stored_once(client):
    nb, _ = await _setup(client)
    r = await client.post(f"/notebooks/{nb}/sources", files={"file": ("again.pdf", PDF)}, headers=A)
    assert r.status_code == 201 and r.json()["created"] is False
    assert len((await client.get(f"/notebooks/{nb}/sources", headers=A)).json()) == 1
