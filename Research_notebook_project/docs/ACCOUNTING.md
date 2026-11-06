# Usage accounting and budget semantics

## Four quantities, never mixed
| Quantity | Source | Where |
|---|---|---|
| **Provider-reported** tokens | `usage` fields returned by the provider (also from a failed attempt, if the error carries them) | ledger `tokens_in/out` where `usage_provenance='reported'` |
| **Estimated** input tokens | local chars/4 estimator | ledger `est_tokens_in`; shown only as "local estimate" |
| **Reserved** tokens | `estimated prompt + max_tokens`, taken before each request | ledger `reserved_tokens` |
| **Budget-accounted** tokens | what the caps charge | `spend_expr()` in `app/core/llm.py` |

`usage_known=false` (`usage_provenance='unknown'`) means the provider reported nothing (timeout, 429 without usage, omitted
usage). `tokens_in/out` are then **0 meaning unknown, not zero**; the attempt is not priced; the budget keeps charging the
full `reserved_tokens`. A DB `CHECK` keeps `unknown <=> usage_known=false`.

## The invariant
For one job, per non-cache ledger row: `charge = tokens_in + tokens_out` if `usage_known` else `reserved_tokens`.
`budget_accounted = sum(charge)` and always equals
`provider_reported + legacy_recorded + unknown_usage_reserved` (tested). Cache hits are excluded everywhere. Phase caps use
the same expression. Reported usage replaces the reservation in both directions (below or above it); the two are never added.

## Lifecycle of one provider attempt
1. `AttemptReserver.begin`: `SELECT ... FOR UPDATE` on the job row, check job and phase caps against everything already charged
   (including other calls' in-flight reservations), insert a `status='reserved'` row, commit (lock released). Over the cap:
   rollback, `BudgetExceededError`, no row, no lock held.
2. Provider request (timeout 60 s).
3. `finish`: the same row becomes `ok`/`failed` with reported usage, or stays unknown with its reservation.
A worker killed between 1 and 3 leaves the `reserved` row, which keeps charging (visible as "interrupted" in reports).
Each failover provider and each schema-retry round is its own attempt, so fallback usage is recorded separately and counted once.

## Concurrency guarantee and limits
- Guarantee: concurrent `call_llm` calls on one job cannot jointly reserve more than the job/phase cap. Verified on PostgreSQL
  (10 simultaneous calls, cap fits 4: exactly 4 start) with a negative control showing the pre-fix design lets all 10 through.
- The lock is per job and held only for check + insert, so there is no multi-lock ordering and no deadlock path.
- **Overshoot is possible after the fact**: if a provider reports more tokens than reserved (estimator undercounts a prompt), the
  charge is the larger reported number. The cap is enforced on reservations, not on provider behavior.
- Conservative cost: failed attempts with unknown usage keep their whole reservation (up to `max_tokens` each), so a 429/timeout
  storm can exhaust a budget on usage that never happened. Chosen deliberately over undercounting.
- The circuit breaker, rate limiter and cache are still in-process / local disk (Phase 4). There is **no idempotency key** yet:
  a crash-resumed node that is not served by the cache will call the provider again and be charged again.

## Historical rows
Migration `003` labels every pre-existing ledger row `usage_provenance='legacy'`. Before 003, a missing provider usage was replaced by a
chars/4 estimate stored in `tokens_in/out`, and the two cases are indistinguishable. Legacy rows are preserved, still counted by the budget
(conservative), but reports show them as "Legacy Recorded Tokens" and never as provider-reported; their modeled USD is shown separately.
Downgrade drops only the added columns and constraints; original token values are untouched (tested on PostgreSQL).

## How the numbers are exposed
`app/core/accounting.py::ledger_summary` is the single aggregation used by `GET /jobs/{id}` (`usage` object; `tokens_spent` is now
**budget-accounted**, `usd_spent` is modeled list price), `GET /jobs/{id}/ledger` (per-attempt `usage_provenance`, `status`, `provider`,
`call_id`, `reserved_tokens`, `est_tokens_in`), the report's section 7, and the MCP `research` result. `scripts/run_eval_jobs.py` exports
reported tokens only plus unknown/failed attempt counts. Modeled USD is list price x provider-reported tokens, never a bill.

## PostgreSQL validation
`tests/pg/` runs only with `TEST_DATABASE_URL` (superuser URL of a throwaway pgvector server; each test creates and drops its own
database). Without it the tests are skipped, which proves nothing about PostgreSQL. Command:
```
docker run -d --name citeguard-pgtest -e POSTGRES_USER=citeguard -e POSTGRES_PASSWORD=citeguard_password -e POSTGRES_DB=citeguard -p 55432:5432 pgvector/pgvector:pg16
TEST_DATABASE_URL=postgresql+asyncpg://citeguard:citeguard_password@localhost:55432/citeguard pytest tests/pg -v
```
