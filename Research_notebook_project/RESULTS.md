# RESULTS

**Status: pipeline cost/latency are measured on real runs; verifier accuracy is measured only on a small synthetic set.** Nothing here is a claim until a row below has a command, a data file hash and a number. Synthetic or author-labeled data is labeled as such and is diagnostic only.

## Methodology
- Positive class = "kept" (verifier says `supports`). **precision** = P(gold supports | kept); **retention** = P(kept | gold supports). Always read together: dropping everything gives high precision and zero retention.
- 95% CIs: percentile bootstrap over claims, 2000 resamples, seed 0 (`scripts/eval_metrics.py`). Claims from one report are correlated, so these CIs are optimistic until grouped resampling is added. When precision is k/k on few claims the bootstrap collapses to [1, 1], so a Wilson interval is reported too.
- Every report records the predictions file SHA-256 and git HEAD.

## Measured

### 1. Verifier on 24 hard cases (live Stage 2)
Data: `evals/hard_cases.csv`, **synthetic, labeled by the author** (9 supports / 2 partial / 13 unsupported); categories: paraphrase, rounding, negation, hedging, multi-hop, overstated, adversarial snapshot. Diagnostic only, not real-world precision.
Command: `DATABASE_URL=... python scripts/eval_hard_cases.py --stage2` then `python scripts/eval_report.py evals/eval_runs/hard_cases_full_predictions.csv`.
Verifier `qwen/qwen3.8-27b` on Groq, cache disabled (0 cache hits), 17 provider calls (7 rows were dropped by Stage 1 before any LLM call), 11,039 provider-reported tokens, $0.0063 modeled.

| Arm | Accuracy (3-class) | Macro-F1 |
|---|---|---|
| Full verifier (Stage 1 + Stage 2) | 0.750 | 0.647 |
| Stage 1 only | 0.417 | 0.287 |
| Best string baseline (claim-vs-span fuzzy, threshold 90/95) | 0.500 | 0.271 |
| Always "unsupported" (label prior) | 0.542 | - |

Kept-claim view (kept = verdict `supports`): precision 5/5 = 1.000 (**Wilson 95% CI 0.566-1.000**; the bootstrap gives a degenerate [1, 1], ignore it), retention 5/9 = 0.556 (Wilson 0.267-0.811). No unsupported claim was kept; the cost is false drops.
Errors (6/24): paraphrase 1 and 3 dropped; rounding 7 ("about 12" vs 12.4) and 9 dropped by Stage 1 fact tokens; partial 6 called unsupported; overstated 21 called partial (would go to rewrite, not shipped as-is).

### 2. Pipeline runs on 10 real questions
Data: `evals/benchmark_questions.json` dev (q1,q5,q6,q9,q12,q13) + test (q2,q8,q10,q14), live Groq + Tavily, shadow mode, fresh cache dir.
Command: `python scripts/run_eval_jobs.py --split dev --yes` (and `--split test`), then `python scripts/summarize_real_runs.py`.
Caveat: q2, q10 and q13 were re-run after two bug fixes (writing-phase budget cap, rate-limiter IndexError), so the 10 jobs span more than one commit; each file's `provenance` records its git commit.

| Metric | Value |
|---|---|
| Jobs completed | 10 / 10 |
| LLM calls | 142 (1 cache hit, 7 failed / unknown-usage attempts) |
| Provider-reported tokens | 189,200 in + 21,095 out |
| Cost | $0.1262 **modeled** from a price table, not billed |
| Per-call latency p50 / p95 | 1,179 ms / 2,342 ms (n = 141 calls; per call, not per report) |
| Claims drafted | 121 (5 failed citation resolution) |
| Shadow verdicts | 65 supports, 12 partial, 39 unsupported, 5 none (**model outputs, not accuracy**) |

Offline arms only (no key needed): `python scripts/eval_hard_cases.py`; analysis in [docs/HARD_CASE_RESULTS.md](docs/HARD_CASE_RESULTS.md).

## Not yet measured (blocked)
| Item | Blocker |
|---|---|
| Verifier precision/retention/F1 with CIs on human labels | label `evals/real/dev_label_sheet.csv` (73) and `test_label_sheet.csv` (43) per docs/ANNOTATION_GUIDELINES.md; 116 total, short of the 150 target |
| Cohen's kappa | second annotator on a subset (`python scripts/cohens_kappa.py`) |
| NLI and LLM-judge baselines | NLI needs ~1.6 GB download; LLM-judge arm not built |
| Retrieval recall@k, hybrid BM25 | gold-source labels; the pipeline does not query embeddings at write time today |
| Ablation A/B/C with tokens, cost, p50/p95 | live provider runs + approval |

## Reproduce the reporter
`python scripts/eval_report.py predictions.csv --out evals/eval_runs/report.json` with columns `id,gold,pred[,score]`. Unit tests: `python -m pytest tests/test_eval_metrics.py`.
