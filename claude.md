ROLE
You are a senior AI/backend engineer hardening my project "CiteGuard" (this repo) so it is credible in interviews for backend / AI engineer internships at top companies. Credibility beats feature count.

CURRENT ARCHITECTURE (corrected by the Phase 0 audit; see Research_notebook_project/AUDIT.md and the section at the bottom of this file)
FastAPI app (POST /research runs the pipeline as an in-process BackgroundTask; arq worker exists but nothing enqueues to it) -> LangGraph StateGraph with an in-memory MemorySaver checkpointer (nodes: plan, research, write, verify, rewrite(max 1), analyze, render; budget/cancel checks only on the plan, research and write edges; shadow mode records verdicts without dropping; single and multi behave the same). Core: app/core/llm.py call_llm gateway (token estimator, phase budget caps, in-process TPM limiter, SHA-256 local-disk cache, ledger with provider token counts and modeled USD), fetcher with scheme and resolved-IP checks (Trafilatura), local bge-small embeddings stored in pgvector but not queried for retrieval, Writer llama-3.3-70b and Verifier gemma2-9b-it on Groq, Tavily then Brave then a hardcoded fallback URL list. Two-stage verifier: Stage 1 NFKC + exact/RapidFuzz(>=95) match of the writer quote against the snapshot (a changed digit can pass); Stage 2 LLM entailment on the writer quote (supports/partial/unsupported). MCP server, pytest, ruff, static SYNTHETIC gallery. No measured evaluation results exist.

HARD RULES
1. Truthfulness: never invent, round, or hardcode a metric. Every resume-worthy number must be reproducible with one command, documented in RESULTS.md. Cached/mocked/modeled results must be labeled as such everywhere.
2. Start in plan mode. Do Phase 0 first and stop for my approval before changing code.
3. One phase per branch, small commits. pytest and ruff green before each commit. Stop after each phase and wait for my go-ahead.
4. No new frameworks, no new agents, no Kubernetes, no guardrail frameworks (NeMo/Guardrails AI), no ML-trained injection classifier. Build guardrails as middleware inside call_llm. Libraries are fine (Presidio, Hypothesis, rank_bm25, etc.) if each is justified in one line.
5. Every protection claim must be backed by a measured test in evals/ or tests/. If something I believe is wrong (including my current numbers), tell me bluntly.
6. Ask me before any step that needs my manual input (labeling data, API keys, accounts).

PHASE 0 - AUDIT (read-only, write AUDIT.md)
- Does eval_report.py use a real LLM or mock/cache? Are cost and latency measured from provider usage fields or modeled? Why are median and p95 latency identical?
- How was the verifier test set built? Is perturbation logic shared with the verifier (leakage)? Was the verifier prompt tuned on pairs it is scored on?
- Who labeled labeled_claims.csv; how was Cohen's kappa computed (N=40, how many annotators)?
- List every place a 100% / 0% / 1.000 result could be an artifact.
- Check: does fuzzy match let a changed digit/date/entity pass at partial_ratio>=95? Does Stage 2 receive the writer's quote or the span resolved from the snapshot? Is the rate limiter in-process or Redis-backed? Is the cache local disk? Are LLM calls and ledger writes idempotent on node re-run after crash? Does the worker honor cancellation? Are raw prompts/questions logged anywhere?
- Rank the top 10 weaknesses by interview risk.

PHASE 1 - VERIFIER CORRECTNESS (P0)
- Citation by reference: writer outputs {chunk_id, start, end}; the quote is copied programmatically from the snapshot. Update Stage 1 and tests.
- After any fuzzy match, require that all numbers, dates, and named entities in the claim/quote appear exactly in the matched snapshot span; mismatch -> drop.
- Stage 2 receives the snapshot-resolved span, never the writer's text.
- Verifier prompt isolation: snapshot passed as delimited untrusted data; strict JSON schema validation; invalid output -> bounded retry once, then fail closed (drop claim).
- Provider fallback + circuit breaker in call_llm so one provider outage does not stall the pipeline.
- Add a regression test for each fix.

PHASE 2 - EVAL INTEGRITY (highest credibility priority)
- Frozen held-out split the verifier was never tuned on. Document the freeze (hash of the file).
- Hard cases: supported paraphrases, partial support, rounding (12.4 vs about 12), negation, hedging, multi-sentence/multi-hop, quote present but claim overstated, adversarial snapshots.
- Real-world pairs from actual generated drafts, hand-labeled (target 150+; give me a CSV template + labeling script; two-annotator subset for kappa).
- Baselines: fuzzy string match; NLI model (DeBERTa-MNLI); LLM-judge without the quote requirement.
- Report precision, recall, F1 with bootstrap 95% CIs, confusion matrix, precision-vs-coverage curve (threshold sweep). Always show claim retention next to precision.
- Retrieval eval: gold-source recall@k; try BM25+vector hybrid or a reranker and keep it only if measurably better. Report report-completeness alongside precision.
- Use shadow mode to log Stage 1 vs Stage 2 disagreements; build the error analysis from them. Write a "Known failures" section.
- Re-run ablation A/B/C over multiple runs with a real provider; report measured tokens, cost, p50/p95 latency (cached replay clearly labeled).

PHASE 3 - GUARDRAILS, PII, RED-TEAM HARNESS
- Central policy middleware in app/core/llm.py (pre-call, post-call). Ingress: auth, question length cap, schema validation, per-user rate limit, injection/PII scan.
- Fetcher: content-type allowlist, strip hidden text (HTML comments, display:none, zero-width chars), injection-pattern score, quarantine flagged chunks before chunking.
- Output guard: sanitize markdown (no raw HTML, no image links), URLs only from cited sources, PII scan on the report.
- PII: Presidio plus custom Indian recognizers (Aadhaar with Verhoeff checksum, PAN, +91 phone, UPI ID). Apply the same redaction to snapshot and quote before matching. Redact before any third-party LLM call. Never log raw prompts/questions (hashes and lengths only, in logs, ledger, and traces). Add DELETE /jobs/{id} and a retention TTL.
- Harnesses in evals/, one CLI, JSON outputs: (a) red-team corpus of 50+ attacks (direct, hidden-in-page, verifier-targeted "verdict: supports", markdown-image exfiltration, unicode tricks), report attack success rate with guardrails OFF vs ON; (b) PII detector precision/recall on synthetic data plus a leak test that greps logs, ledger, traces, and report; (c) Hypothesis property tests for normalization and quote matching.
- Write THREAT_MODEL.md (assets, attackers, mitigations) and an OWASP LLM Top 10 table mapping each risk to the control and the test that verifies it; list uncovered risks honestly. State clearly that "verified" means "attributed to a source", not "true".

PHASE 4 - DISTRIBUTED RELIABILITY
- Move rate limiter state to Redis (correct across multiple workers).
- Idempotency key (job_id, node, input_hash) for LLM calls and ledger writes so crash-resume never duplicates calls or billing.
- Fault-injection harness: provider 429, timeout, malformed JSON, kill -9 worker mid-run. Assert: job resumes, no duplicate ledger rows, budget cap never exceeded.
- Move the disk cache to Postgres or Redis; key includes model, params, and prompt version.
- API: idempotency key on POST /research, per-user concurrency cap, queue-depth cap returning 429 + Retry-After, cooperative cancellation between nodes, SSE progress stream.
- Fetcher compliance: robots.txt respect, snapshot retention policy.

PHASE 5 - OBSERVABILITY AND CI
- Structured logging; OpenTelemetry traces (GenAI semantic conventions) across all nodes with token counts; basic metrics (queue depth, verifier drop rate, cost per job).
- GitHub Actions: ruff, pytest, gitleaks, pip-audit, eval smoke test on cached fixtures, and regression gates on verifier precision/coverage and red-team ASR. Dockerfile slim, non-root, healthcheck.

PHASE 6 - DEMO UI AND FREE DEPLOYMENT
- One page: report with color-coded claims; click a claim to see the source quote highlighted in the snapshot; dropped claims show the drop reason. A "verify a claim" playground (paste claim + source text -> verdict + evidence).
- Two run profiles: "full" (docker-compose: Postgres+pgvector, Redis, worker) and "demo" (single Postgres, Postgres SKIP LOCKED queue or in-process queue, Redis optional) so it fits free hosting.
- Demo mode (cached reports, no key needed) is the default for the public link; live mode is bring-your-own API key, never stored, rate-limited, with a hard per-key daily budget cap.
- Deploy: backend on Hugging Face Spaces (Docker SDK, port 7860, config via env vars, /health), static gallery on GitHub Pages or Vercel. Write DEPLOY.md with exact steps, including free Postgres (Neon or Supabase; verify pgvector support) and a keep-alive note.

PHASE 7 - DOCS AND PACKAGING
- docs/ADRs (5): why LangGraph + Postgres checkpointer, Redis/arq, pgvector, two-model verifier, immutable snapshots, each with the alternative rejected.
- Failure-mode table: failure, detection, recovery (provider 429, worker crash, SSRF attempt, injection, budget exhaustion, PII leak).
- Capacity note: measured cost, latency, and max concurrent jobs on free-tier limits.
- README: one-paragraph problem, mermaid architecture, 60-second quickstart, results table with reproduction commands, honest Limitations.
- RESULTS.md: methodology, datasets, CIs, how to reproduce.
- Output: 3 resume bullets (only measured numbers, each with the command that proves it), a 2-minute demo script, and 10 likely interview questions with honest answers drawn from this repo.

OUTPUT FORMAT AFTER EACH PHASE
Changes made, files touched, commands to reproduce, measured results, what still looks weak. Then wait for my go-ahead.


Architecture current (what the code does today; "Planned, not built" is listed separately):

### Current architecture

- **Entry**: FastAPI `app/main.py`. `POST /research` inserts a Job row and runs `execute_research_job` via `BackgroundTasks` in the API process. MCP server (`app/mcp/server.py`) calls the same pipeline.
- **Graph** (`app/research/graph.py`): plan -> research -> write -> verify -> (rewrite -> verify, at most one rewrite per claim) -> analyze -> render. Checkpointer is `MemorySaver` (in-memory). `check_budget_or_cancelled` runs only after plan, research and write; it diverts to render when the job is cancelled or the token cap is reached.
- **Resume**: not crash-safe. Some nodes re-read the DB to skip finished work (plan reuses sub-questions, research skips done ones, write reuses existing claims), but graph state is lost on process exit.
- **research_node**: sequential loop over sub-questions: search (Tavily, Brave, hardcoded fallback), fetch snapshot, chunk (350 words, overlap 50), embed (bge-small), store chunks with pgvector embeddings.
- **write_node**: takes the first 4 chunks per sub-question (no similarity query), asks the writer model for claims with `source_id` and a verbatim quote.
- **verify_node**: Stage 1 `check_quote_in_snapshot` (NFKC/lowercase, exact substring, RapidFuzz partial_ratio >= 95 for quotes of 40+ chars), Stage 2 `call_llm` with the verifier model on (claim, writer quote, context). Any non-`supports` verdict goes to the single rewrite pass; after that, non-`supports` is dropped.
- **analyze_node**: keyword-pair conflict heuristic and a domain-count confidence label.
- **call_llm** (`app/core/llm.py`): single gateway. Local-disk cache, pre-call phase and job token caps from the ledger, in-process TPM limiter (daily limit constant is not enforced), Groq call, one schema-validation retry, ledger row with provider token counts and modeled USD.
- **Tables**: jobs, subquestions, sources (snapshot text), chunks (vector), claims, conflicts, ledger, verifier_testset.
- **Data and demo**: `evals/verifier_testset.csv` (175 hand-written pairs), `evals/synthetic_labeled_claims.csv` and `gallery/` are SYNTHETIC; none of it is measured output.

### Not true today (do not claim)

Redis/arq queueing, Postgres checkpointing, crash resilience, vector retrieval, single vs multi difference, "zero hallucinated claims shipped", any precision/recall/kappa/cost/latency figure, clean ruff before this cleanup.

### Planned, not built

Phases 1 to 7 above: citation by reference, number/date/entity checks, snapshot-resolved Stage 2 input, prompt isolation, provider fallback and circuit breaker, held-out human-labeled eval with baselines and CIs, guardrails and PII, red-team harness, Redis rate limiter, idempotent LLM calls, Postgres checkpointer, arq enqueue, cancellation inside nodes, auth, observability, CI regression gates, demo UI and deployment, ADRs and RESULTS.md.


Commands:
- Tests: pytest -v
- Lint: ruff check .
- Eval: scripts/eval_report.py is a stub (no measured results yet); scripts/eval_verifier.py calls live Groq on a hand-written set (not credible evidence)