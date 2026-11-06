# Phase 2 research audit (Stage 1)

Date: 2026-11-06. Branch `phase2-eval-integrity`, HEAD `cc202a9`. Read-only inspection; the only change in this stage is this file.
Test state at HEAD: 122 passed, 13 skipped (the skips are `tests/pg/`, which need PostgreSQL; **PG-specific behaviour is unverified here**). `ruff check .` clean.

## 1. Pipeline as it exists

plan -> research (search -> fetch snapshot -> chunk 350w/50 -> bge-small embed -> pgvector store) -> write (first 4 chunks per sub-question, **no similarity query**) -> verify (Stage 1 deterministic: offsets, number/date/entity tokens present in span; Stage 2 LLM entailment on the snapshot span) -> rewrite (max 1, `partial` only) -> analyze -> render.
Eval entry points: `scripts/eval_verifier.py` (synthetic CSV, calls `verify_claim`), `scripts/run_eval_jobs.py` (real jobs, dry-run by default, `--yes` to spend), `make_label_sheet.py`, `freeze_dataset.py`, `cohens_kappa.py`, `nli_baseline.py`, `check_source_overlap.py`.

## 2. Status

| Item | Status |
|---|---|
| Citation by reference, fact-token check, span-only Stage 2, fail-closed | Done, unit tested (mocked LLM) |
| Usage accounting from provider fields | Done, tested |
| Real-draft export, blind label sheet, labeling guide, freeze, kappa | Tooling only. `evals/real/` is **empty**: zero real drafts, zero human labels |
| Hard cases (24, synthetic, single author) | Created, **never evaluated** |
| Any measured verifier number | **None.** The only run artifact is the degenerate fuzzy baseline (see 3.1) |
| Retrieval recall@k, ablation A/B/C, statistics, splits by source | Not built |

## 3. Scientific-validity risks (ranked)

1. **The fuzzy baseline is degenerate.** `evals/eval_runs/fuzzy_gemma2-9b-it.json`: `kept_precision 0.0`, `retention 0.0`, `false_alarm 1.0`. `fuzzy_baseline` compares the *claim* to the quote with `partial_ratio >= 95`; paraphrased claims almost never reach 95, so it says "unsupported" for everything. The reported precision 0.79 / recall 0.99 is just the always-"unsupported" rate. It is a strawman and must not be used as the "baseline to beat". A fair string baseline should match against the span with a lower, tuned-on-dev threshold, or use the NLI baseline.
2. **No true held-out data.** `is_heldout` is a SHA-1 hash split of the same 175 rows, and `prompt_iterations.md` states the prompt was tuned on all 175. The "heldout" column is not held out. Treat the 175-row set as dev/diagnostic only.
3. **Leakage by construction in the synthetic set.** Each row is an original quote plus a programmatic perturbation (number/date/entity/overgeneralize/causal) of the *same* sentence. The perturbation types mirror what Stage 1 checks, so Stage 1 is evaluated on the failure modes it was written for. Hash-by-quote keeps perturbations of one quote in one split (good), but the set has no paraphrase, negation or hedging, i.e. nothing Stage 2 is supposed to be good at.
4. **Three-class labels are collapsed.** `eval_verifier.report` is built around "unsupported" as the positive class. `partial` is folded into negatives; no macro-F1, no per-class P/R. Labels in the 175-set are only `supports`/`unsupported` (36/139), so the binary framing is also majority-class dominated (79% unsupported).
5. **CI covers only recall.** `bootstrap_ci` resamples only positives for recall. No CI on precision, retention or kept-claim precision; with n=40 held-out and recall 1.0 the interval is [1.0, 1.0], which says nothing.
6. **Retrieval is not what the README-style description suggests.** Embeddings are stored but never queried; `write_node` takes the first 4 chunks. So "Recall@k of the existing retrieval strategy" has two defensible meanings: (a) web-search source recall (`search_web`, `top_n=3`, Tavily -> Brave -> hardcoded fallback), and (b) chunk-level recall of an embedding query that **does not exist yet** and would be new code. Both need gold relevance labels that do not exist. The hardcoded fallback list can inflate (a) if gold sources are in it.
7. **No second annotator, no real labels.** Kappa is undefined. Any kappa reported now would be fabricated.
8. **Single synthetic annotator for the hard cases**, and the author of the cases also wrote the verifier prompt guidance. Labels such as "partial" for the bridge case are judgment calls (e.g. "longest in the country" is *unsupported* addition by the labeling guide's own rule "adds an unproven detail" -> `partial`; the case is labeled `partial`, consistent, but `partial` vs `unsupported` is exactly where annotators will disagree). Diagnostic use only.
9. **Cache/mock contamination risk.** `call_llm` has a SHA-256 disk cache; the earlier "100%" came from a pre-filled answer-key cache (since deleted). `run_eval_jobs.py` points the cache to a fresh dir, but `eval_verifier.py` does **not**, so repeated or stale runs can silently replay. Any eval must record cache hit counts or disable the cache.
10. **Non-determinism and model drift.** Groq model ids can change behind the same name; temperature/seed are not recorded in run files (only `prompt_sha`). Runs are not reproducible from the JSON alone (no git hash, dataset hash, timestamp, usage).
11. **Stage 1 hard failures are scored as predictions.** Unresolvable citations count as "dropped (unsupported)" in `eval_verifier`. That is correct for a pipeline metric but conflates citation failure with entailment errors; report separately.
12. **`mode` single/multi/shadow** behave the same (documented), so there is no real "single vs multi" ablation to run.

## 4. What can be ablated (real toggles only)

Nothing is a flag today. Components that exist and could be switched on in eval code without new features:
- Stage 1 fact-token check (call `locate`/skip `missing_fact_tokens`).
- Stage 2 LLM entailment on span vs. on writer quote vs. no-span (judge without quote requirement).
- Rewrite pass on/off (pipeline-level only; needs real drafts).
Non-existent: reranker, hybrid retrieval, vector retrieval at write time. Ablations A/B/C must be defined from the first group only (proposal in the ablation doc, Stage 5). Do not invent others.

## 5. Engineering risks

- PG tests skipped here; migration 003 and idempotency untested on PG in this environment.
- Budget/cancel checks only on plan/research/write edges; a verify-heavy run can overshoot the cap by one node.
- `run_eval_jobs.py` lacks provider/model selection, resume, per-run provenance (git hash, dataset hash, timestamps), cost estimate before run beyond the dry-run plan, failure log.
- Many scripts (`generate_*`, `seed_*`) produce or load synthetic data; none is labeled in its output consistently except `eval_verifier`.

## 6. Plan and dependencies

| Stage | Work | Blocked on |
|---|---|---|
| 2 | Annotation guidelines, template, validator, import, agreement | none; then **human labels** |
| 3 | Run 24 hard cases: Stage 1 deterministic (offline) + fuzzy; Stage 2 LLM | Stage 2 needs Groq key + approval; NLI needs ~1.6 GB download (optional extra) |
| 4 | IR metrics code + tests | gold relevance labels (human) |
| 5 | Ablation runner on A/B/C defined in section 4 | Groq key for LLM arms |
| 6-7 | Real labels, second annotator, freeze by source | **human** |
| 8 | Real-provider runs | Tavily + Groq keys, explicit cost approval |
| 9-10 | Stats, figures, report | labels + results |

No paid call has been made and none is planned without your approval.
