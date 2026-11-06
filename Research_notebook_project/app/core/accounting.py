"""Ledger aggregates shared by the API, the report and the MCP tool, so every surface agrees with the budget.

Four different quantities, never added together:
  provider_reported_*   rows whose usage came from the provider (usage_provenance='reported')
  legacy_recorded_*     rows written before migration 003: provider usage OR an old local estimate, unverifiable
  estimated_input_tokens local chars/4 estimate of prompt size (never provider usage)
  budget_accounted_tokens what the budget charges: reported/legacy tokens where counted, else the retained reservation
Cache hits are excluded from all of them (they cost the provider nothing).
"""
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm import spend_expr
from app.models import LedgerEntry

_L = LedgerEntry


def _sum(expr, *conds):
    return func.coalesce(func.sum(expr).filter(*conds), 0)


def _count(*conds):
    return func.count().filter(*conds)


async def ledger_summary(db: AsyncSession, scope_type: str, scope_id: int) -> dict[str, Any]:
    prov = _L.cache_hit.is_(False)
    rep = (prov, _L.usage_provenance == "reported")
    leg = (prov, _L.usage_provenance == "legacy")
    unk = (prov, _L.usage_provenance == "unknown")
    row = (await db.execute(
        select(
            _sum(_L.tokens_in, *rep).label("rep_in"),
            _sum(_L.tokens_out, *rep).label("rep_out"),
            _sum(_L.cost_usd_modeled, *rep).label("rep_usd"),
            _sum(_L.tokens_in, *leg).label("leg_in"),
            _sum(_L.tokens_out, *leg).label("leg_out"),
            _sum(_L.cost_usd_modeled, *leg).label("leg_usd"),
            _count(*unk).label("unk_n"),
            _sum(_L.reserved_tokens, *unk).label("unk_reserved"),
            _sum(_L.est_tokens_in, prov).label("est_in"),
            _sum(spend_expr(), prov).label("budget"),
            _count(prov).label("attempts"),
            _count(prov, _L.status == "failed").label("failed"),
            _count(prov, _L.status == "reserved").label("in_flight"),
            _sum(_L.latency_ms, prov).label("latency_ms"),
            _count(_L.cache_hit.is_(True)).label("cache_hits"),
        ).where(_L.scope_type == scope_type, _L.scope_id == scope_id)
    )).one()
    by_model = (await db.execute(
        select(_L.provider, _L.model, func.count(), func.sum(_L.tokens_in + _L.tokens_out).filter(_L.usage_provenance == "reported"))
        .where(_L.scope_type == scope_type, _L.scope_id == scope_id, _L.cache_hit.is_(False))
        .group_by(_L.provider, _L.model).order_by(_L.provider, _L.model)
    )).all()
    return {
        "provider_reported_tokens_in": int(row.rep_in),
        "provider_reported_tokens_out": int(row.rep_out),
        "provider_reported_tokens": int(row.rep_in + row.rep_out),
        "legacy_recorded_tokens": int(row.leg_in + row.leg_out),  # provenance unverifiable (pre-migration 003)
        "unknown_usage_attempts": int(row.unk_n),
        "unknown_usage_reserved_tokens": int(row.unk_reserved),  # charged to the budget, not known to be spent
        "estimated_input_tokens": int(row.est_in),  # local estimate, NOT provider usage
        "budget_accounted_tokens": int(row.budget),
        "provider_attempts": int(row.attempts),
        "failed_attempts": int(row.failed),
        "in_flight_or_interrupted_attempts": int(row.in_flight),
        "cache_hits": int(row.cache_hits),
        "modeled_usd_reported": float(row.rep_usd),  # modeled at list price from provider-reported tokens only
        "modeled_usd_legacy": float(row.leg_usd),
        "latency_ms": int(row.latency_ms),
        "attempts_by_provider_model": [
            {"provider": p, "model": m, "attempts": int(n), "provider_reported_tokens": int(t or 0)} for p, m, n, t in by_model
        ],
    }
