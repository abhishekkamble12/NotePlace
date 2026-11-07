# CiteGuard

**A research pipeline that refuses to ship a claim it can't trace to a source.** You ask a question. CiteGuard
splits it into sub-questions, searches the web, and stores immutable page snapshots. It retrieves the most relevant
chunks with pgvector, has an LLM draft claims that cite exact snapshot offsets, and runs every claim through a
two-stage verifier before it can appear in the report. Every LLM call goes through one gateway that enforces a token
budget and writes a ledger row.

> "Verified" means *attributed to a span of a fetched page*, not *true*. If a page is wrong, CiteGuard will faithfully
> cite the wrong page.

**Stack:** Python 3.12, FastAPI, LangGraph, PostgreSQL + pgvector, SQLAlchemy (async) + Alembic,
sentence-transformers (bge-small), Groq (writer `openai/gpt-oss-120b`, verifier `qwen/qwen3.8-27b`), Tavily/Brave
search, MCP server, pytest + Hypothesis + mutmut.

---

## How it works

```mermaid
flowchart TD
    Q([POST /research or MCP tool]) --> Plan[plan: LLM splits the question into sub-questions]
    Plan --> Research[research: search, SSRF-safe fetch, snapshot,<br/>chunk with offsets, embed bge-small into pgvector]
    Research --> Write[write: top-4 chunks per sub-question by cosine distance,<br/>LLM drafts claims citing chunk + sentence offsets]
    Write --> Verify[verify: Stage 1 deterministic, Stage 2 LLM entailment]
    Verify -->|partial, at most once| Rewrite[rewrite against the snapshot span]
    Rewrite --> Verify
    Verify --> Analyze[analyze: cross-source conflicts, source-diversity confidence]
    Analyze --> Render[render: 7-section Markdown report with audit trail]
    Plan -. budget or cancel .-> Render
    Research -. budget or cancel .-> Render
    Write -. budget or cancel .-> Render
```

### The verifier (`app/core/verifier.py`)

1. **Citation by reference.** The writer cites `chunk_id` + sentence indices, never a quote. Code copies the span out
   of the stored snapshot, so the writer cannot misquote a source.
2. **Stage 1, deterministic, 0 tokens.** The offsets must be valid, and every number, date and named entity in the
   claim must appear in the cited span. "33.9%" in the source and "43.9%" in the claim is a drop
   (`fact_token_mismatch`) before any LLM is called.
3. **Stage 2, LLM entailment.** A different model family judges `supports | partial | unsupported` from the span
   only. It never sees the writer's text. The span is fenced as untrusted data with an HMAC-derived delimiter that
   page text cannot forge. Output is schema-validated and retried once. If it's still invalid, the claim is dropped.
4. **Fail closed.** Provider outage, exhausted budget, invalid output or a missing key all drop the claim with a
   reason code. None of them can let a claim through.

### Engineering details worth asking about

- **Token budget with reservations.** `call_llm` locks the job row, checks job and per-phase caps against everything
  already charged (including in-flight calls), and writes a `reserved` ledger row before the request. Provider-reported
  usage replaces the reservation afterwards. Details: [docs/ACCOUNTING.md](docs/ACCOUNTING.md).
- **Provider fallback and circuit breaker** in the same gateway, so one provider outage doesn't stall a job.
- **SSRF-hardened fetcher.** Scheme check, and the resolved IP is checked against private, loopback, link-local and
  metadata ranges. Every redirect hop is validated *before* it is contacted. The connection is pinned to the vetted IP
  (no second DNS lookup, so no rebinding). A text-only content-type allowlist and a 2 MB cap are enforced while
  streaming.
- **API hygiene.** Bounded inputs (question length, budget range), admin key on the export and label-import endpoints
  (fails closed if unset), CSV formula-injection escaping, and failed-job reports that never echo exception text.
- **MCP server.** `python -m app.mcp.server` exposes `research` and `verify_quote` to any MCP client.

---

## Testing

All counts below are from runs on this branch. Reproduce with the commands shown.

| What | Result | Command |
|---|---|---|
| Unit, property, integration (real Postgres) and API tests | 331 passed | `pytest` (with `TEST_DATABASE_URL` set) |
| Fault injection: real `call_llm` + Groq SDK against a scriptable fake provider (429s, timeouts, malformed JSON) | 13 passed, 1 known gap (below) | `make test-chaos` |
| Branch coverage on `verifier.py`, `llm.py`, `fetcher.py` | 88.2% (CI gate 85%) | `make test-cov` |
| Mutation score on `verifier.py` (mutmut, 570 mutants) | 87.0% killed | `make test-mutation` |
| Security suite: SSRF matrix against real local servers, 37-payload prompt-injection corpus | included above | `make test-sec` |

Property tests (Hypothesis) found a real normalization bug, now fixed with a regression test: normalizing `İ`
followed by a combining mark wasn't idempotent, so a quote copied from such a snapshot couldn't be found again. Mutation testing found that `rewrite_claim` had no tests at all. While fixing that, I found that the rewrite
step was sending the writer's quote instead of the snapshot span. Both are fixed.

---

## Measured results

Full methodology and reproduction steps: [RESULTS.md](RESULTS.md).

| Measurement | Value | Data |
|---|---|---|
| Real pipeline runs | 10/10 jobs completed, 142 LLM calls, $0.126 total (modeled from list prices), per-call latency p50 1.18 s / p95 2.34 s | 10 live questions, Groq + Tavily |
| Verifier on hard cases, full vs Stage 1 only vs best string baseline | 3-class accuracy 0.750 vs 0.417 vs 0.500 | 24 **synthetic, author-written** cases, so diagnostic only |
| Precision / recall on real claims with human labels | **not measured yet** | label sheets exist for 116 real claims, pending independent labeling |

---

## Quickstart

Prerequisites: Python 3.12+, Docker, a Groq API key, and a Tavily or Brave API key. Without a search key, sub-questions
get no sources; the pipeline never invents them.

```bash
cd Research_notebook_project
cp .env.example .env              # add GROQ_API_KEY, TAVILY_API_KEY, CITEGUARD_EVIDENCE_HMAC_KEY
docker compose up -d              # Postgres + pgvector on :5432
uv sync --extra test
uv run alembic upgrade head
uv run uvicorn app.main:app --port 8000 --env-file .env
```

```bash
curl -X POST localhost:8000/research -H "Content-Type: application/json" \
     -d '{"question": "What efficiency records have perovskite-silicon tandem solar cells reached?", "max_tokens": 40000}'
curl localhost:8000/jobs/<job_id>            # status, claims kept, tokens and modeled cost
curl localhost:8000/jobs/<job_id>/report     # the Markdown report
```

Tests: `TEST_DATABASE_URL=postgresql+asyncpg://citeguard:citeguard_password@localhost:5432/citeguard uv run pytest`.

---

## Limitations

- **Single process.** Jobs run as FastAPI background tasks with an in-memory LangGraph checkpointer, so a restart loses
  in-flight jobs. There's no queue and no multi-worker rate limiter.
- **The budget can be exceeded by one call's estimation error.** Caps are checked against a chars/4 estimate before the
  call. If the provider reports more usage afterwards, it's recorded, and the overshoot is pinned by an explicit
  `xfail` test.
- **Stage 1 is strict by design.** "About 12" vs 12.4, entity aliases (Google/Alphabet) and acronyms cause false drops.
  Negation and hedging rely entirely on Stage 2. See [docs/VERIFIER_LIMITATIONS.md](docs/VERIFIER_LIMITATIONS.md).
- **Retrieval hasn't been benchmarked.** Top-4 by cosine distance replaced "first 4 chunks", and there's no recall@k
  measurement yet.
- **Auth covers only the admin endpoints.** Job endpoints are open, so this isn't meant to face the internet as-is.
