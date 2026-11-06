"""ledger_summary / report totals reconcile with the budget model for every ledger shape (SQLite; PostgreSQL in tests/pg)."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.accounting import ledger_summary
from app.models import Job, LedgerEntry, Source, SubQuestion
from app.reporter import generate_markdown_report

BASE = dict(scope_type="job", scope_id=1, phase="writing", kind="llm")


def reported(tin, tout, *, status="ok", provider="groq", model="m", usd=0.001, reserved=1000, call_id="c1"):
    return LedgerEntry(**BASE, tokens_in=tin, tokens_out=tout, usage_known=True, usage_provenance="reported", status=status,
                       provider=provider, model=model, cost_usd_modeled=usd, reserved_tokens=reserved, est_tokens_in=40,
                       latency_ms=100, call_id=call_id)


def unknown(*, status="failed", provider="groq", model="m", reserved=1000, call_id="c1"):
    return LedgerEntry(**BASE, tokens_in=0, tokens_out=0, usage_known=False, usage_provenance="unknown", status=status,
                       provider=provider, model=model, reserved_tokens=reserved, est_tokens_in=40, latency_ms=50, call_id=call_id)


def legacy(tin, tout, usd=0.002):
    return LedgerEntry(**BASE, tokens_in=tin, tokens_out=tout, usage_known=True, usage_provenance="legacy", cost_usd_modeled=usd)


def cache_hit():
    return LedgerEntry(**BASE, tokens_in=0, tokens_out=0, usage_known=True, usage_provenance="reported", cache_hit=True)


class Adapter:
    def __init__(self, s):
        self.s = s

    async def execute(self, stmt):
        return self.s.execute(stmt)


async def summarize(rows):
    eng = create_engine("sqlite://")
    LedgerEntry.__table__.create(eng)
    with Session(eng) as s:
        s.add_all(rows)
        s.commit()
        return await ledger_summary(Adapter(s), "job", 1)


def check_identity(u):
    assert u["budget_accounted_tokens"] == (
        u["provider_reported_tokens"] + u["legacy_recorded_tokens"] + u["unknown_usage_reserved_tokens"]
    )


@pytest.mark.asyncio
async def test_known_usage():
    u = await summarize([reported(100, 50), reported(10, 5)])
    assert (u["provider_reported_tokens"], u["budget_accounted_tokens"], u["unknown_usage_attempts"]) == (165, 165, 0)
    assert u["estimated_input_tokens"] == 80  # estimate reported separately, never added to actual
    check_identity(u)


@pytest.mark.asyncio
async def test_unknown_usage_is_labelled_and_charged_its_reservation_only():
    u = await summarize([reported(100, 50), unknown(reserved=700)])
    assert u["provider_reported_tokens"] == 150 and u["unknown_usage_attempts"] == 1
    assert u["unknown_usage_reserved_tokens"] == 700 and u["budget_accounted_tokens"] == 850
    assert u["modeled_usd_reported"] == pytest.approx(0.001)  # unknown attempt is not priced
    check_identity(u)


@pytest.mark.asyncio
async def test_provider_failure_with_reported_usage_and_fallback_success():
    u = await summarize([
        reported(30, 0, status="failed", call_id="c1"),
        reported(20, 5, provider="openai_compat", model="m2", call_id="c1"),
    ])
    assert (u["provider_attempts"], u["failed_attempts"], u["provider_reported_tokens"]) == (2, 1, 55)
    assert [(m["provider"], m["provider_reported_tokens"]) for m in u["attempts_by_provider_model"]] == [("groq", 30), ("openai_compat", 25)]
    check_identity(u)


@pytest.mark.asyncio
async def test_multiple_attempts_in_one_call_each_counted_once():
    u = await summarize([reported(10, 2, call_id="c1"), reported(12, 3, call_id="c1"), unknown(status="reserved", call_id="c2")])
    assert u["provider_attempts"] == 3 and u["in_flight_or_interrupted_attempts"] == 1
    assert u["provider_reported_tokens"] == 27 and u["budget_accounted_tokens"] == 27 + 1000
    check_identity(u)


@pytest.mark.asyncio
async def test_cache_hits_are_excluded_from_spend_but_counted():
    u = await summarize([reported(10, 5), cache_hit(), cache_hit()])
    assert u["cache_hits"] == 2 and u["provider_attempts"] == 1 and u["budget_accounted_tokens"] == 15
    check_identity(u)


@pytest.mark.asyncio
async def test_historical_rows_are_separate_from_provider_reported():
    u = await summarize([legacy(700, 50), reported(10, 5)])
    assert u["provider_reported_tokens"] == 15  # legacy never counted as provider-reported
    assert u["legacy_recorded_tokens"] == 750 and u["modeled_usd_legacy"] == pytest.approx(0.002)
    assert u["budget_accounted_tokens"] == 765  # but the budget still counts them (conservative)
    check_identity(u)


@pytest.mark.asyncio
async def test_empty_ledger():
    u = await summarize([])
    assert u["budget_accounted_tokens"] == 0 and u["attempts_by_provider_model"] == []


@pytest.mark.asyncio
async def test_report_text_labels_each_quantity():
    usage = await summarize([reported(100, 50), unknown(reserved=700), legacy(300, 20)])
    db = AsyncMock()
    db.get.return_value = Job(id=1, question="Q?", status="done", max_tokens=10_000)

    def res(items):
        r = MagicMock()
        r.scalars.return_value.all.return_value = items
        return r

    db.execute.side_effect = [res([SubQuestion(id=1, job_id=1, text="s", status="done")]), res([]),
                              res([Source(id=1, job_id=1, url="u", domain="d", snapshot="x")]), res([])]
    with patch("app.reporter.ledger_summary", AsyncMock(return_value=usage)):
        md = await generate_markdown_report(db, job_id=1)
    assert "Provider-Reported Tokens:** 150 (100 input, 50 output)" in md
    assert "Legacy Recorded Tokens:** 320" in md and "may include local estimates" in md
    assert "Attempts With Unknown Usage:** 1" in md and "700 tokens reserved" in md
    assert "Estimated Input Tokens (local estimate, not provider usage)" in md
    assert "Token Budget Accounted:** 1,170 / 10,000" in md
