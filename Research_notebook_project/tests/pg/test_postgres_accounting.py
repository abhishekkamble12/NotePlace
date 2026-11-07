"""PostgreSQL integration tests for migrations, ledger constraints, reservations and concurrency.

Opt-in: set TEST_DATABASE_URL to a SUPERUSER URL of a throwaway server (pgvector image), e.g.
  docker run -d --name citeguard-pgtest -e POSTGRES_USER=citeguard -e POSTGRES_PASSWORD=citeguard_password \
      -e POSTGRES_DB=citeguard -p 55432:5432 pgvector/pgvector:pg16
  TEST_DATABASE_URL=postgresql+asyncpg://citeguard:citeguard_password@localhost:55432/citeguard pytest tests/pg -v
Every test creates and drops its own database, so nothing in the server's other databases is touched.
Without TEST_DATABASE_URL these tests are skipped; a skipped run proves nothing about PostgreSQL.
"""
import asyncio
import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from alembic.config import Config
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from alembic import command
from app.core import llm
from app.core.accounting import ledger_summary
from app.core.llm import BudgetExceededError, ProviderUnavailableError, call_llm, spend_expr
from app.models import Job, LedgerEntry

BASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not BASE_URL, reason="TEST_DATABASE_URL not set: PostgreSQL integration NOT validated")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _admin_url():
    return BASE_URL.rsplit("/", 1)[0] + "/postgres"


def _run(coro):
    return asyncio.run(coro)


async def _admin(sql):
    eng = create_async_engine(_admin_url(), isolation_level="AUTOCOMMIT", poolclass=NullPool)
    async with eng.connect() as c:
        await c.execute(text(sql))
    await eng.dispose()


def _alembic(url, action, rev, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", url)  # alembic/env.py reads this at run time
    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "alembic"))
    getattr(command, action)(cfg, rev)


@pytest.fixture
def newdb(monkeypatch):
    name = "ct_" + uuid.uuid4().hex[:10]
    _run(_admin(f'CREATE DATABASE "{name}"'))
    url = BASE_URL.rsplit("/", 1)[0] + "/" + name
    yield url, monkeypatch
    _run(_admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))


async def _query(url, sql, **params):
    eng = create_async_engine(url, poolclass=NullPool)
    async with eng.connect() as c:
        res = await c.execute(text(sql), params)
        rows = res.all()
    await eng.dispose()
    return rows


# ---------------------------------------------------------------- schema / migrations
def test_fresh_database_schema(newdb):
    url, mp = newdb
    _alembic(url, "upgrade", "head", mp)
    cols = {r[0]: r for r in _run(_query(url, "SELECT column_name, data_type, is_nullable, column_default FROM information_schema.columns WHERE table_name='ledger'"))}
    for name, dtype in [("usage_known", "boolean"), ("reserved_tokens", "integer"), ("est_tokens_in", "integer"),
                        ("status", "character varying"), ("provider", "character varying"),
                        ("call_id", "character varying"), ("usage_provenance", "character varying")]:
        assert cols[name][1] == dtype and cols[name][2] == "NO", name
    assert cols["usage_provenance"][3] is None  # no default: every new row must state provenance
    assert "true" in cols["usage_known"][3] and "0" in cols["reserved_tokens"][3]
    checks = {r[0] for r in _run(_query(url, "SELECT conname FROM pg_constraint WHERE conrelid='ledger'::regclass AND contype='c'"))}
    assert {"ck_ledger_provenance_known", "ck_ledger_provenance_values"} <= checks
    idx = {r[0] for r in _run(_query(url, "SELECT indexname FROM pg_indexes WHERE tablename='ledger'"))}
    assert "ix_ledger_scope" in idx
    fks = _run(_query(url, "SELECT conname FROM pg_constraint WHERE contype='f' AND conrelid IN ('claims'::regclass, 'chunks'::regclass, 'subquestions'::regclass)"))
    assert len(fks) >= 3 and any("chunk" in r[0] for r in fks)


def test_migration_from_002_labels_old_rows_legacy_and_is_reversible(newdb):
    url, mp = newdb
    _alembic(url, "upgrade", "002_citation_by_reference", mp)
    _run(_query(url, "SELECT 1"))

    async def seed():
        eng = create_async_engine(url, poolclass=NullPool)
        async with eng.begin() as c:
            await c.execute(text("INSERT INTO jobs (id, question, max_tokens, max_minutes, mode, status, stage, report_md) VALUES (1,'q',50000,8,'standard','done','rendering','')"))
            await c.execute(text("INSERT INTO ledger (scope_type, scope_id, phase, kind, model, tokens_in, tokens_out, cost_usd_modeled, latency_ms, cache_hit) VALUES ('job',1,'writing','llm','m',700,50,0.001,10,false),('job',1,'writing','llm','m',0,0,0,0,true)"))
        await eng.dispose()
    _run(seed())
    _alembic(url, "upgrade", "head", mp)
    rows = _run(_query(url, "SELECT tokens_in, tokens_out, usage_provenance, usage_known, reserved_tokens, status FROM ledger ORDER BY id"))
    assert [tuple(r) for r in rows] == [(700, 50, "legacy", True, 0, "ok"), (0, 0, "legacy", True, 0, "ok")]
    _alembic(url, "downgrade", "002_citation_by_reference", mp)
    old = _run(_query(url, "SELECT tokens_in, tokens_out FROM ledger ORDER BY id"))
    assert [tuple(r) for r in old] == [(700, 50), (0, 0)]  # historical values survive the round trip
    cols = {r[0] for r in _run(_query(url, "SELECT column_name FROM information_schema.columns WHERE table_name='ledger'"))}
    assert "usage_provenance" not in cols and "usage_known" not in cols
    _alembic(url, "upgrade", "head", mp)  # and re-applies cleanly


# ---------------------------------------------------------------- DB-backed fixtures
@pytest.fixture
def sessions(newdb, monkeypatch, tmp_path):
    url, mp = newdb
    _alembic(url, "upgrade", "head", mp)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm.rate_limiter, "acquire", AsyncMock())
    monkeypatch.delenv("CITEGUARD_FALLBACKS", raising=False)
    llm.breaker.reset()
    holder = {}

    async def make():
        eng = create_async_engine(url, poolclass=NullPool)
        holder["eng"] = eng
        return async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False, autoflush=False)
    return make, holder


async def _mk_job(sm, max_tokens=10_000):
    async with sm() as s:
        j = Job(question="q", max_tokens=max_tokens)
        s.add(j)
        await s.commit()
        return j.id


async def _ledger(sm, job_id):
    async with sm() as s:
        return (await s.execute(select(LedgerEntry).where(LedgerEntry.scope_id == job_id).order_by(LedgerEntry.id))).scalars().all()


async def _call(sm, job_id, tag, provider, phase="research", max_tokens=1000):
    async with sm() as s:
        with patch("app.core.llm._call_provider", provider):
            return await call_llm(scope=("job", job_id), phase=phase, model="m",
                                  messages=[{"role": "user", "content": tag}], max_tokens=max_tokens, db=s)


# ---------------------------------------------------------------- constraints and accounting
@pytest.mark.asyncio
async def test_provenance_check_constraint_rejects_inconsistent_rows(sessions):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)
    async with sm() as s:
        s.add(LedgerEntry(scope_type="job", scope_id=jid, phase="research", usage_known=False, usage_provenance="reported"))
        with pytest.raises(IntegrityError):
            await s.commit()
    async with sm() as s:
        s.add(LedgerEntry(scope_type="job", scope_id=jid, phase="research", usage_known=True, usage_provenance="bogus"))
        with pytest.raises(IntegrityError):
            await s.commit()


@pytest.mark.asyncio
async def test_failed_attempt_unknown_usage_and_fallback_rows(sessions, monkeypatch):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)
    monkeypatch.setenv("CITEGUARD_FALLBACKS", '{"m": [{"provider": "x", "model": "m2", "base_url": "http://x"}]}')

    class E(Exception):
        usage = {"prompt_tokens": 40, "completion_tokens": 0}

    out = await _call(sm, jid, "fb", AsyncMock(side_effect=[E(), ("fine", 20, 5)]))
    assert out == "fine"
    primary, fallback = await _ledger(sm, jid)
    assert (primary.provider, primary.status, primary.usage_provenance, primary.tokens_in) == ("groq", "failed", "reported", 40)
    assert (fallback.provider, fallback.status, fallback.tokens_in, fallback.tokens_out) == ("x", "ok", 20, 5)
    assert primary.call_id == fallback.call_id
    async with sm() as s:
        u = await ledger_summary(s, "job", jid)
    assert u["provider_reported_tokens"] == 65 and u["budget_accounted_tokens"] == 65  # one count per attempt, no double counting
    assert u["provider_attempts"] == 2 and u["failed_attempts"] == 1
    assert {(m["provider"], m["model"]) for m in u["attempts_by_provider_model"]} == {("groq", "m"), ("x", "m2")}


@pytest.mark.asyncio
async def test_all_providers_timeout_retains_reservations(sessions, monkeypatch):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)
    monkeypatch.setattr(llm, "PROVIDER_TIMEOUT_S", 0.05)

    async def hang(*a, **k):
        await asyncio.sleep(5)

    with pytest.raises(ProviderUnavailableError):
        await _call(sm, jid, "to", hang)
    (row,) = await _ledger(sm, jid)
    assert (row.status, row.usage_provenance, row.usage_known, row.tokens_in) == ("failed", "unknown", False, 0)
    async with sm() as s:
        u = await ledger_summary(s, "job", jid)
    assert u["provider_reported_tokens"] == 0 and u["unknown_usage_attempts"] == 1
    assert u["budget_accounted_tokens"] == row.reserved_tokens > 0  # not zero: unknown is not free


@pytest.mark.asyncio
async def test_actual_below_and_above_reservation(sessions):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)
    await _call(sm, jid, "below", AsyncMock(return_value=("a", 10, 5)), max_tokens=1000)
    await _call(sm, jid, "above", AsyncMock(return_value=("b", 4000, 3000)), max_tokens=100)  # provider used more than reserved
    below, above = await _ledger(sm, jid)
    assert below.reserved_tokens > 15 and above.reserved_tokens < 7000
    async with sm() as s:
        u = await ledger_summary(s, "job", jid)
    assert u["budget_accounted_tokens"] == 15 + 7000  # reported replaces the reservation in both directions


@pytest.mark.asyncio
async def test_interrupted_call_keeps_reservation(sessions):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)

    async def killed(*a, **k):
        raise asyncio.CancelledError()  # stands in for a worker killed mid-request

    with pytest.raises(asyncio.CancelledError):
        await _call(sm, jid, "kill", killed)
    (row,) = await _ledger(sm, jid)
    assert row.status == "reserved" and row.usage_known is False
    async with sm() as s:
        u = await ledger_summary(s, "job", jid)
    assert u["in_flight_or_interrupted_attempts"] == 1 and u["budget_accounted_tokens"] == row.reserved_tokens


@pytest.mark.asyncio
async def test_budget_exceeded_rolls_back_and_releases_lock(sessions):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm, max_tokens=1_000)  # research cap 500 < reservation (~1010)
    with pytest.raises(BudgetExceededError):
        await _call(sm, jid, "toobig", AsyncMock(return_value=("x", 1, 1)))
    assert await _ledger(sm, jid) == []
    async with sm() as s:  # lock must be free: this would block forever if the failed call kept it
        await asyncio.wait_for(s.get(Job, jid, with_for_update=True), 5)


@pytest.mark.asyncio
async def test_identical_request_after_success_is_a_cache_hit_not_a_second_charge(sessions):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)
    p = AsyncMock(return_value=("same", 10, 5))
    await _call(sm, jid, "dup", p)
    await _call(sm, jid, "dup", p)
    assert p.await_count == 1
    async with sm() as s:
        u = await ledger_summary(s, "job", jid)
    assert u["provider_reported_tokens"] == 15 and u["cache_hits"] == 1


# ---------------------------------------------------------------- concurrency
@pytest.mark.asyncio
async def test_concurrent_calls_cannot_overspend_the_phase_cap(sessions):
    """10 simultaneous calls, each reserving ~1010 tokens, research cap 5000 => at most 4 may start."""
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm, max_tokens=10_000)
    started = 0

    async def slow(*a, **k):
        nonlocal started
        started += 1
        await asyncio.sleep(0.5)  # all reservations are in flight together
        return "ok", 500, 100

    results = await asyncio.gather(*[_call(sm, jid, f"c{i}", slow) for i in range(10)], return_exceptions=True)
    ok = [r for r in results if r == "ok"]
    refused = [r for r in results if isinstance(r, BudgetExceededError)]
    assert len(ok) == 4 and len(refused) == 6 and started == 4, (len(ok), len(refused), results)
    async with sm() as s:
        total = (await s.execute(select(func.sum(spend_expr())))).scalar()
    assert total == 4 * 600 <= 5000  # in flight they held 4*~1010 <= cap; settled they hold the reported 600 each


@pytest.mark.asyncio
async def test_unknown_usage_failures_exhaust_budget_conservatively(sessions):
    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm, max_tokens=10_000)  # research cap 5000 => 4 reservations of ~1010 fit
    outcomes = []
    for i in range(6):
        llm.breaker.reset()  # otherwise the circuit opens after 3 failures and later calls make no attempt at all
        try:
            await _call(sm, jid, f"u{i}", AsyncMock(side_effect=TimeoutError()))
        except ProviderUnavailableError:
            outcomes.append("provider")
        except BudgetExceededError:
            outcomes.append("budget")
    assert outcomes == ["provider"] * 4 + ["budget"] * 2  # failures with unknown usage keep charging their reservation
    async with sm() as s:
        assert (await ledger_summary(s, "job", jid))["budget_accounted_tokens"] <= 5000


@pytest.mark.asyncio
async def test_api_endpoints_report_the_same_numbers_as_the_ledger(sessions):
    from app.main import get_job_ledger, get_job_status

    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm)
    await _call(sm, jid, "api-ok", AsyncMock(return_value=("a", 100, 20)))
    with pytest.raises(ProviderUnavailableError):
        await _call(sm, jid, "api-fail", AsyncMock(side_effect=TimeoutError()))
    async with sm() as s:
        status = await get_job_status(jid, s)
        ledger = await get_job_ledger(jid, s)
        summary = await ledger_summary(s, "job", jid)
    assert status["usage"] == summary
    assert status["tokens_spent"] == summary["budget_accounted_tokens"] == 120 + ledger[1]["reserved_tokens"]
    assert [(e["status"], e["usage_provenance"], e["tokens_in"]) for e in ledger] == [("ok", "reported", 100), ("failed", "unknown", 0)]


@pytest.mark.asyncio
async def test_control_without_reservation_rows_the_cap_is_overspent(sessions):
    """Negative control: the pre-fix design (check, then call, write the row afterwards) lets all 10 calls through.
    This proves the concurrency test above can detect the race, not merely that it passes."""
    from app.core.llm import AttemptReserver, check_budget_caps

    async def old_begin(self, provider, model, reserved, est_in):
        await check_budget_caps(self.session, self.scope, self.phase, reserved)  # no lock, nothing committed
        row = LedgerEntry(scope_type=self.scope[0], scope_id=self.scope[1], phase=self.phase, kind="llm", model=model,
                          usage_known=False, usage_provenance="unknown", reserved_tokens=reserved, est_tokens_in=est_in,
                          status="reserved", provider=provider, call_id=self.call_id)
        self.session.add(row)
        return row

    make, _ = sessions
    sm = await make()
    jid = await _mk_job(sm, max_tokens=10_000)
    started = 0

    async def slow(*a, **k):
        nonlocal started
        started += 1
        await asyncio.sleep(0.5)
        return "ok", 500, 100

    with patch.object(AttemptReserver, "begin", old_begin):
        await asyncio.gather(*[_call(sm, jid, f"x{i}", slow) for i in range(10)], return_exceptions=True)
    assert started == 10  # 10 x ~1010 reserved against a 5000 cap: the race is real without the fix
