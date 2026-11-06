"""Provider-failure accounting: reported vs unknown vs reserved usage, fallback attempts, budget reconciliation."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.core import llm
from app.core.llm import ProviderUnavailableError, call_llm, spend_expr
from app.models import Job, LedgerEntry


class UsageError(Exception):
    def __init__(self, usage=None):
        super().__init__("provider failed")
        self.usage = usage


@pytest.fixture(autouse=True)
def _fresh_breaker(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)  # never read or pollute the real disk cache
    llm.breaker.reset()
    llm.rate_limiter.reset()
    monkeypatch.delenv("CITEGUARD_FALLBACKS", raising=False)
    yield
    llm.breaker.reset()


def make_db():
    db = AsyncMock()
    db.add = MagicMock()
    db.get.return_value = Job(id=1, max_tokens=1_000_000)
    row = MagicMock(total_tokens=0, phase_tokens=0)
    db.execute.return_value = MagicMock(one=MagicMock(return_value=row))
    return db


def rows(db):
    return [c.args[0] for c in db.add.call_args_list]


async def run(db, tag, provider):
    with patch("app.core.llm._call_provider", provider):
        return await call_llm(scope=("job", 1), phase="writing", model="m",
                              messages=[{"role": "user", "content": tag}], max_tokens=100, db=db)


@pytest.mark.asyncio
async def test_reported_usage_on_failed_attempt_is_recorded():
    db = make_db()
    with pytest.raises(ProviderUnavailableError):
        await run(db, "t-fail-reported", AsyncMock(side_effect=UsageError({"prompt_tokens": 70, "completion_tokens": 9})))
    (r,) = rows(db)
    assert (r.status, r.usage_known, r.tokens_in, r.tokens_out) == ("failed", True, 70, 9)
    assert r.reserved_tokens > 0 and r.cost_usd_modeled > 0


@pytest.mark.asyncio
async def test_unknown_usage_is_unknown_not_estimated():
    db = make_db()
    # failed attempt with no usage, and a success whose provider omitted usage
    calls = AsyncMock(side_effect=[TimeoutError(), ("ok", None, None)])
    with patch.dict("os.environ", {"CITEGUARD_FALLBACKS": json.dumps({"m": [{"provider": "x", "model": "m2", "base_url": "http://x"}]})}):
        assert await run(db, "t-unknown", calls) == "ok"
    failed, ok = rows(db)
    for r in (failed, ok):
        assert r.usage_known is False and r.tokens_in == 0 and r.tokens_out == 0
        assert r.cost_usd_modeled == 0.0  # unknown usage is not priced
        assert r.reserved_tokens > 0 and r.est_tokens_in > 0  # estimate kept in its own column only


@pytest.mark.asyncio
async def test_fallback_success_records_both_attempts_once_each():
    db = make_db()
    calls = AsyncMock(side_effect=[UsageError({"prompt_tokens": 10, "completion_tokens": 0}), ("fine", 20, 5)])
    with patch.dict("os.environ", {"CITEGUARD_FALLBACKS": json.dumps({"m": [{"provider": "x", "model": "m2", "base_url": "http://x"}]})}):
        assert await run(db, "t-fallback", calls) == "fine"
    primary, fallback = rows(db)
    assert (primary.provider, primary.model, primary.status, primary.tokens_in) == ("groq", "m", "failed", 10)
    assert (fallback.provider, fallback.model, fallback.status, fallback.tokens_in, fallback.tokens_out) == ("x", "m2", "ok", 20, 5)
    assert primary.call_id == fallback.call_id and primary.call_id
    # no double counting: the call-level total is exactly the sum of the two attempt rows
    assert sum(r.tokens_in + r.tokens_out for r in rows(db)) == 35


@pytest.mark.asyncio
async def test_successful_call_one_row_reported_usage():
    db = make_db()
    assert await run(db, "t-ok", AsyncMock(return_value=("hi", 12, 3))) == "hi"
    (r,) = rows(db)
    assert (r.status, r.usage_known, r.tokens_in, r.tokens_out) == ("ok", True, 12, 3)


def test_budget_reconciliation_reported_replaces_reservation_unknown_keeps_it():
    eng = create_engine("sqlite://")
    LedgerEntry.__table__.create(eng)
    base = dict(scope_type="job", scope_id=1, phase="writing")
    with Session(eng) as s:
        s.add_all([
            # known: charged its reported 150, NOT its 1000 reservation, and not both
            LedgerEntry(**base, tokens_in=100, tokens_out=50, reserved_tokens=1000, usage_known=True),
            # unknown (e.g. timeout): full reservation retained
            LedgerEntry(**base, tokens_in=0, tokens_out=0, reserved_tokens=700, usage_known=False, usage_provenance="unknown", status="failed"),
            # cache hit: excluded
            LedgerEntry(**base, tokens_in=0, tokens_out=0, reserved_tokens=0, cache_hit=True),
        ])
        s.commit()
        total = s.execute(select(func.sum(spend_expr())).where(LedgerEntry.cache_hit.is_(False))).scalar()
    assert total == 150 + 700
