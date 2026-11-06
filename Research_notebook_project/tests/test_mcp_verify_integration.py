# tests/test_mcp_verify_integration.py
"""Integration test of the REAL verify_quote handler. verify_claim, call_llm, the breaker and the ledger logic all run.
Replaced (honestly labelled): the database session (in-memory FakeSession, NOT PostgreSQL) and the provider HTTP call
(_call_provider returns canned JSON). Real PostgreSQL behaviour is covered separately (migration task), not here."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

import app.core.llm as llm
from app.mcp.server import VERIFY_QUOTE_MAX_TOKENS, verify_quote
from app.models import Job, LedgerEntry

SNAP = ("Background text. In late 2023 the Acme lab reported a peak efficiency of 33.9% for the new perovskite cell, "
        "measured by Fraunhofer ISE in Freiburg. More text follows here.")
QUOTE = SNAP[SNAP.index("In late"): SNAP.index("Freiburg.") + len("Freiburg.")]


class FakeSession:
    """In-memory stand-in for AsyncSession. Sums ledger rows in Python instead of SQL."""
    store: list = []

    def __init__(self):
        self.store = FakeSession.store

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    def add(self, obj):
        if isinstance(obj, Job) and obj.id is None:
            obj.id = 1 + sum(isinstance(o, Job) for o in self.store)
        self.store.append(obj)

    async def commit(self):
        return None

    async def refresh(self, obj):
        return None

    async def get(self, model, pk, **_):
        return next((o for o in self.store if isinstance(o, model) and o.id == pk), None)

    async def execute(self, stmt):
        ledger = [o for o in self.store if isinstance(o, LedgerEntry) and not o.cache_hit]
        total = sum(o.tokens_in + o.tokens_out for o in ledger)
        return SimpleNamespace(one=lambda: SimpleNamespace(total_tokens=total, phase_tokens=total))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.delenv("CITEGUARD_FALLBACKS", raising=False)
    llm.breaker.reset()
    llm.rate_limiter.reset()
    FakeSession.store = []
    provider = AsyncMock(return_value=('{"verdict": "supports", "reason": "entailed"}', 120, 30))
    with patch("app.mcp.server.init_db", AsyncMock()), \
         patch("app.mcp.server.AsyncSessionLocal", FakeSession), \
         patch.object(llm, "_call_provider", provider):
        yield provider
    llm.breaker.reset()


async def _run(**kw):
    return json.loads(await verify_quote(**kw))


@pytest.mark.asyncio
async def test_real_handler_supports_with_context_creates_job_and_ledger(env):
    out = await _run(claim="Acme reported 33.9% efficiency in late 2023.", quote=QUOTE, context=SNAP)

    assert out["action"] == "kept" and out["entailment_verdict"] == "supports" and out["quote_found"] is True
    assert out["source_authenticated"] is False
    assert out["span_source"] == "caller_supplied_context"
    jobs = [o for o in FakeSession.store if isinstance(o, Job)]
    assert len(jobs) == 1 and jobs[0].id == out["job_id"]
    assert jobs[0].max_tokens == VERIFY_QUOTE_MAX_TOKENS and jobs[0].status == "done"
    ledger = [o for o in FakeSession.store if isinstance(o, LedgerEntry)]
    assert len(ledger) == 1 and ledger[0].scope_id == out["job_id"]
    assert ledger[0].tokens_in == 120 and ledger[0].tokens_out == 30  # provider-reported usage, not estimates
    assert env.await_count == 1


@pytest.mark.asyncio
async def test_no_context_is_labelled_as_caller_supplied_quotation(env):
    out = await _run(claim="Acme reported 33.9% efficiency in late 2023.", quote=QUOTE)
    assert out["span_source"] == "caller_supplied_quotation"
    assert out["source_authenticated"] is False


@pytest.mark.asyncio
async def test_changed_number_is_dropped_before_any_provider_call(env):
    out = await _run(claim="Acme reported 43.9% efficiency in late 2023.", quote=QUOTE, context=SNAP)
    assert out["action"] == "dropped" and out["drop_reason"] == "fact_token_mismatch"
    env.assert_not_awaited()


@pytest.mark.asyncio
async def test_quote_absent_from_context_is_dropped(env):
    out = await _run(claim="Acme reported 33.9% efficiency.", quote="Completely different sentence about cats and dogs.", context=SNAP)
    assert out["action"] == "dropped" and out["drop_reason"] == "quote_not_found"
    env.assert_not_awaited()


@pytest.mark.asyncio
async def test_provider_outage_fails_closed_through_real_handler(env):
    env.side_effect = RuntimeError("503")
    out = await _run(claim="Acme reported 33.9% efficiency in late 2023.", quote=QUOTE, context=SNAP)
    assert out["action"] == "dropped" and out["drop_reason"] == "provider_unavailable"
    assert [o for o in FakeSession.store if isinstance(o, Job)][0].status == "done"


@pytest.mark.asyncio
async def test_response_keeps_existing_fields_for_consumers(env):
    out = await _run(claim="Acme reported 33.9% efficiency in late 2023.", quote=QUOTE, context=SNAP)
    for key in ("quote_found", "match_score", "entailment_verdict", "action", "reason"):
        assert key in out
