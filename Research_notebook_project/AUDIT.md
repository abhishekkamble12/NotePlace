# CiteGuard — Phase 0 Audit (read-only)

Date: 2026-11-06. Scope: `Research_notebook_project/`. No code was changed.
How checked: read every file under app/, scripts/, evals/, tests/, CI; ran `pytest` (23 passed), `ruff check .` (3 errors), `python scripts/eval_report.py`, and probed `check_quote_in_snapshot` directly.

## TL;DR (blunt)
Every headline number in the README (100% precision, 100% recall, 0% false-drop, 45.2% → 0.0%, κ = 0.904, 54.8% retention, 100% budget adherence, cost and latency, the Config A/B/C ablation) comes from **hardcoded or synthesized data**, not from runs of the system. None of them can go on a resume. The architecture description is also wrong in four load-bearing places: no Redis queue in use, no Postgres checkpointer, no vector retrieval, no single/multi difference.

---

## 1. eval_report.py: real LLM or mock/cache?
**Neither. It runs no model at all.** It reads `evals/labeled_claims.csv`, which `scripts/generate_labeled_benchmark.py` writes from a Python literal (`RAW_BENCHMARK_DATA`):
- Verifier outputs (`found`, `entail`, `action`) are typed by hand into the literal (generate_labeled_benchmark.py:29-407). They are not verifier outputs.
- The rows are the same ~65 claims **copied 3×** (`multiplier = 3`, lines 491-511) to reach 195 "claims".
- `tokens_used` = 2840 if kept else 3120; `cost_usd_modeled` = 0.0028/0.0031; `latency_seconds` = **18.4 constant** for B/C and 32.1 for A (lines 455-486).
- **Why median = p95 latency:** each job's latency is that single constant, so every job has 18.4 s → p50 = p95 = 0.31 min.
- Cost is "modeled" even in the live path: `calculate_modeled_usd` uses a hardcoded price table (app/core/llm.py:34-39). Tokens in the *live* path come from `response.usage` (llm.py:328-330); that part is real. eval_report never touches it.
- Ablation costs and latencies are **multipliers on Config C**: A = `med_cost*0.75`, `med_latency*1.8`; B = `*0.85`, `*0.8` (eval_report.py:177-179).
- `budget_adherence = 100.0` is a literal (eval_report.py:136).
- Fallback literals if the data is empty: `[42500]`, `[0.042]`, `[4.1]` (lines 124-126).
- Config B precision is printed as `100 - unsupported_rate`. That is true only because the dataset has no "partial" label.

## 2. Verifier test set (`evals/verifier_testset.csv`, 175 rows)
- **Built by hand** in `scripts/generate_testset.py`: 35 quotes × 5 claims. 36 supports, 139 unsupported, **0 partial**. Perturbations are blatant (e.g. "Roscosmos conducted the DART mission", "cured all human genetic diseases"). There are no hard cases: no paraphrase, negation, hedging, rounding or multi-hop.
- **Leakage, worst case:** `scripts/cache_benchmark_responses.py` writes the verifier's *answers* into the LLM disk cache, **derived from the gold label and perturbation_type** (lines 57-82). CI runs it before `eval_verifier.py` (`.github/workflows/ci.yml`). Cache keys match exactly (same messages, schema, max_tokens=250), so `call_llm` returns the fabricated verdict and never calls Groq. The "subtle edge case" branch `i in (18, 55)` never fires because rows 18 and 55 are labeled unsupported. The cache is therefore a perfect answer key. `evals/eval_runs/d92fada7_gemma2-9b-it.json` (recall 1.0, precision 1.0, CI [1.0, 1.0]) is exactly what this cache produces. Nothing in the repo shows it came from a live model.
- **Prompt tuned on the scored pairs:** `evals/prompt_iterations.md` says v1→v3 were each scored "on 175 pairs". The v3 prompt lists exactly the test set's perturbation categories (numbers/dates/entities/causation/"some→all"). The `is_heldout` hash split was added afterwards, so the "heldout" slice is not held out from prompt design.
- v1/v2 scores (78.4%, 91.3%, …) have **no run artifacts**. They cannot be verified.
- **Stage 1 is never evaluated:** eval passes `snapshot = context + "\n" + quote` (eval_verifier.py:104), so the quote is always an exact substring.
- `evals/baseline.json` = {recall 1.0, false_alarm 0.0}. The CI `--check` gate is set against the fabricated result.
- Shared perturbation logic with the verifier code: none, but the cache script shares the *labels*, which is worse.
- Model risk: Groq has deprecated `gemma2-9b-it` (verify on the Groq console). If so, a live run fails, and the cache is the only path that "works".

## 3. labeled_claims.csv: who labeled it, and how was κ computed?
- **Nobody labeled it.** `annotator_1_label` is typed into the script. `annotator_2_label` for the 40 "doubly labeled" rows is generated: **a copy of annotator 1, except every 7th claim_id is flipped** (generate_labeled_benchmark.py:499-503).
- κ code (`calculate_cohens_kappa`, eval_report.py:10) is a correct 2-rater Cohen's κ. The input is synthetic, so **κ = 0.904 measures a `% 7` rule, not human agreement.** N = 40, "2 annotators", 0 humans.
- The claims are not from generated drafts. `source_url`s are plausible-looking and were never fetched. Quotes are invented paraphrases of real papers.

## 4. Every 100% / 0% / 1.000 that is (or could be) an artifact
| Result | Where | Why it's an artifact |
|---|---|---|
| Verifier recall/precision 100%, false-alarm 0%, CI [1.0,1.0], F1 1.000 | eval_runs/*.json, prompt_iterations.md, baseline.json | Answer-key cache derived from labels; prompt tuned on same set |
| Citation precision 100%, post-verifier unsupported 0% | eval_report / README | Hand-typed `action="kept"` only on rows hand-typed `supported` |
| Catch rate 100%, false-drop 0% | eval_report / README | Same: verifier column = label column |
| Budget adherence 100% | eval_report.py:136 | Literal |
| Pre-verifier 45.2% / retention 54.8% | eval_report | Ratio of hand-written faithful/perturbed claims; complements by construction |
| κ 0.904 | eval_report | Synthetic annotator 2 |
| p50 = p95 latency | eval_report | Constant 18.4 s |
| Gallery "100% immune to rate limits", per-report tokens/cost/latency | scripts/generate_gallery.py | 10 reports hand-authored with literal metrics (e.g. `tokens_used: 18450`). Not cached real runs |
| "100% clean Ruff" | README:201 | False today: 3 errors in app/main.py (I001, 2× F811) |
| "Zero hallucinated claims shipped" (CLAUDE.md invariant 3) | — | False: see §5, a changed digit/date passes Stage 1 |
| Test `test_verifier_benchmark_testset_integrity` | tests/test_citeguard.py:93 | Checks file shape only, not verifier quality |

## 5. Specific checks
**Fuzzy match lets changed digits/dates pass. CONFIRMED.** Probed `check_quote_in_snapshot` against snapshot "…efficiency of 33.9% in late 2023…":
- quote with `43.9%` → **(True, 99.07)**
- quote with `2013` → **(True, 99.07)**
- inserted "did not" → (False, 94.8) (rejected only narrowly)
- Google→IBM entity swap in a ~100-char quote → (False, 94.1). A longer quote would pass.
The only guard is `MIN_FUZZY_QUOTE_CHARS = 40`. A one-character change in any quote over ~40 chars clears 95.

**Stage 2 receives the writer's quote, not the snapshot span.** `verify_claim` puts the `quote` argument (= `claim.quote` from the writer) into the prompt (verifier.py:143-150). Also:
- `extract_surrounding_context` finds the index in the *normalized* snapshot and slices the *raw* snapshot with it (verifier.py:87-91). The offsets drift.
- After a fuzzy match, `find` returns -1, so the context becomes the **first 800 chars of the page**, unrelated to the quote.
- Snapshot text and claim are not delimited as untrusted. An injected "verdict: supports" in the page reaches the verifier.

**Rate limiter: in-process.** Module-global `SlidingWindowRateLimiter` (llm.py:122). Per process, per model, **TPM only**: `GROQ_TPD_LIMIT` is declared but never enforced. It records *estimates*, not actual usage. It holds one `asyncio.Lock` while sleeping, so every caller (all models) is serialized behind one waiter.

**Cache: local disk.** `~/.cache/citeguard/llm/<sha256>.json` (llm.py:56). The key is model + messages + schema name + max_tokens. No temperature is set. The prompt version is implicit in the message text. The cache is not shared across workers or hosts. Write errors are silently swallowed. Corrupted entries return None silently.

**Idempotency on node re-run after crash: partial, by accident.**
- The checkpointer is **`MemorySaver` (in-memory)**, not Postgres (graph.py:10,136). A worker crash loses all graph state. "Crash-resilience" in the architecture doc is false.
- Resume works partly because nodes re-read the DB: `plan_node` reuses existing SubQuestions, `research_node` skips `done` subqs, `write_node` returns existing claims.
- An LLM call that finished before the crash is a disk-cache hit on re-run, so there is no second provider call. A **second ledger row** (cache_hit=True, 0 tokens) is written each time.
- Cache write happens **before** the ledger write (llm.py:353 vs 373). A crash between the two gives a cache hit on retry, and the real tokens are **never billed**. The budget under-counts.
- `research_node` commits Source rows then Chunks per URL. A crash mid-URL re-inserts chunks on retry (Source dedupes by hash, Chunk does not).
- `verify_node` re-verifies **all** claims on every rewrite loop pass. Cache hits keep that cheap, but it writes extra ledger rows.

**Cancellation: partially honored.** `POST /jobs/{id}/cancel` sets status. `check_budget_or_cancelled` sees it only on the plan→, research→, and write→ edges. Inside a node (e.g. the research loop or verify loop) nothing checks it. verify→rewrite→analyze→render never check. `render_node` then still writes a report.

**Raw prompts/questions logged?** No logger exists. Two `print()`s log URLs and exception text (nodes.py:171, fetcher.py:163). Exceptions from Groq may echo request content. Questions are stored **in plaintext** in `jobs.question` and in `report_md`. On failure, `report_md = "Error: {e}"` (pipeline.py:54) stores raw exception text. The `/export/claims.csv` endpoint returns every job's question with **no auth**. The ledger stores no prompt text (good).

## 6. Architecture claims vs code
| Claimed | Actual |
|---|---|
| POST /research → Redis (arq) → worker | `main.py` uses FastAPI `BackgroundTasks` in the API process (main.py:83). **Nothing enqueues to arq.** The compose `worker` service runs idle |
| LangGraph Postgres checkpointer, thread_id=job_id | `MemorySaver`. thread_id is set but state dies with the process |
| pgvector retrieval (top-k chunks per subq) | Embeddings are computed and stored, then **never queried**. `write_node` takes `.limit(4)` chunks with no `ORDER BY` similarity (nodes.py:231-235) |
| modes single \| multi \| shadow; "parallel fan-out" | No code branches on single vs multi. Research is a sequential `for` loop. **Config A vs B can't differ.** Only shadow changes behavior |
| partial → rewrite; unsupported → drop | `verify_node` sends **every non-"supports" verdict, including unsupported and quote-not-found, to rewrite** (nodes.py:307-312). A rewrite can flip an unsupported claim to kept |
| Tavily search | Tavily → Brave → a hardcoded keyword→URL fallback / Wikipedia guesses (search.py:103). Runs without a key give fake "search" results silently |
| SSRF-safe fetcher | DNS is checked, then httpx resolves again (rebinding TOCTOU). `follow_redirects=True` fetches intermediate hops before only the **final** URL is validated. The 2 MB cap is checked **after** the full body is downloaded. No content-type check |
| Chunks with "injection guards" | Chunks are wrapped in `<evidence_data>` tags. The writer is not told to treat them as data. The verifier gets the raw snapshot with no tags at all |
| 23 tests, ruff clean | 23 pass. Ruff has 3 errors. The tests cover the limiter, budget math, cache, routing, and a mocked pipeline. Nothing tests verifier correctness on hard cases, crash-resume, or cancellation |
| Writer source_id | Not checked against the chunks it was shown. A wrong id gives an empty snapshot and a drop (fails closed, but silently loses good claims) |

## 7. Top 10 weaknesses ranked by interview risk
1. **Fabricated evaluation numbers.** labeled_claims.csv, κ, cost, latency and ablation are hardcoded or synthesized, and the README presents them as human-verified results. An interviewer opening `generate_labeled_benchmark.py` ends the conversation. *Must be removed or relabeled before anything else.*
2. **Verifier "100%" comes from an answer-key cache** built from gold labels (cache_benchmark_responses.py). CI "verifies" it against a baseline of 1.0.
3. **Stage 1 passes altered numbers and dates** (43.9% vs 33.9% → 99.07). This contradicts the core product claim "mutated numbers/dates are dropped".
4. **Stage 2 judges the writer's text, not the source span.** Its context is mis-sliced and falls back to the first 800 chars of the page after a fuzzy match. The verifier prompt has no injection isolation.
5. **Architecture misdescribed:** no arq enqueue, an in-memory checkpointer, no vector retrieval, single==multi. Each is one "show me the code" question away.
6. **Unsupported claims get rewritten and can be kept**, contrary to the documented flow. Quote-not-found claims burn a rewrite call.
7. **Test set is easy and was tuned on:** 175 blatant perturbations, 0 partial labels, prompt iterated on the scored set, the held-out split added after the fact, Stage 1 bypassed.
8. **Crash/billing correctness:** cache-before-ledger loses billing on a crash, duplicate ledger rows on re-run, non-idempotent chunk inserts, an in-process limiter (wrong with >1 worker, TPD unenforced, global lock).
9. **No auth anywhere.** `/export/claims.csv` dumps all questions. `/labels` lets anyone overwrite labels. There are no per-user limits. Questions sit in plaintext with no retention policy.
10. **SSRF gaps** (redirect hops unchecked, DNS rebinding, 2 MB cap enforced after download), silent fake search fallback, and gemma2-9b-it likely deprecated, so live runs may not work at all.

## 8. What is real and worth keeping
- `call_llm` as a single gateway: real usage tokens, pre-call budget check, ledger row per call, schema validation with one retry.
- NFKC normalization plus an exact-substring check (the exact path is sound).
- SSRF IP classification (`is_ip_allowed`) and scheme checks.
- DB-level resume shortcuts in plan, research and write nodes.
- Correct Cohen's κ and bootstrap code, ready for real data.
- `/export/claims.csv` and `/labels` already exist as a labeling loop (add auth).

## 9. Recommended immediate action (before Phase 1, needs your OK)
Strike or relabel every README/gallery metric as "synthetic placeholder". Delete or quarantine `cache_benchmark_responses.py` and its CI step, and reset `baseline.json`. Until Phase 2 produces real numbers, the honest resume line is architectural, not metric.
