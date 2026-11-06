# Hard-case results (offline arms only)

**Data:** `evals/hard_cases.csv`, 24 rows, **synthetic, single annotator (the author), not human-validated**. Gold: 9 supports, 2 partial, 13 unsupported. These are diagnostics, not real-world precision, and no CI is given (n=24, would be misleading).
**Reproduce:** `python scripts/eval_hard_cases.py` -> `evals/eval_runs/hard_cases_offline.json` (records dataset SHA-256 and git HEAD). No API key, no LLM.
**Not run:** Stage 2 (LLM) and the NLI baseline. They are pending credentials/approval and a ~1.6 GB download. Nothing below says anything about the full verifier.
Labels were not changed to fit any arm.

## Arms
- `stage1`: the deterministic check (`locate_quote` + `missing_fact_tokens` of the claim vs the span). Cannot output `partial`; "pass" is scored as `supports`, so this is Stage 1 alone, not the product.
- `fuzzy-N`: claim vs span `partial_ratio >= N`, thresholds swept {60,70,80,90,95}, none picked. This replaces the old degenerate claim-vs-quote baseline.

| Arm | Accuracy | Macro-F1 | Weighted-F1 |
|---|---|---|---|
| stage1 | 0.417 | 0.287 | 0.390 |
| fuzzy-60 | 0.375 | 0.182 | 0.205 |
| fuzzy-70 | 0.292 | 0.202 | 0.275 |
| fuzzy-80 | 0.375 | 0.239 | 0.350 |
| fuzzy-90 / 95 | 0.500 | 0.271 | 0.413 |

Chance-level context: always-"unsupported" gets 13/24 = 0.542 accuracy. Fuzzy-90/95 (0.500) are at or below that, i.e. they carry no information beyond the label prior here.

## Stage 1 errors (14 of 24), by cause
Confusion (gold rows x predicted cols): supports 6/0/3, partial 2/0/0, unsupported 9/0/4.

| Cause | IDs | Meaning |
|---|---|---|
| False drops from number form | 3 ("one in ten" vs "10%"), 7 ("about 12" vs 12.4), 9 ("just over 1,000" vs 1,046) | Documented in `docs/VERIFIER_LIMITATIONS.md` (rounding/word numbers). Costs coverage. |
| Semantic errors Stage 1 cannot see | 11, 12 (negation), 14, 16 (hedging), 18 (multi-hop dates), 19-21 (overstatement), 4 (omitted side effect) | 9 unsupported claims pass because every number/entity is present. These are Stage 2's job; Stage 1 alone does not claim to handle them. |
| `partial` unreachable | 5, 6 | Stage 1 has no partial output. |

Stage 1 got all 3 adversarial-snapshot rows right, but only because the injected claims change a number (`$50M`, `600`); it does not read instructions at all. That is not evidence about prompt-injection resistance of Stage 2.

## Fuzzy errors
Fuzzy match can't separate paraphrase (0/3 at 80) from wrong-number claims (rounding 8 and adversarial 22/23 pass at 80 because the wrong number is a tiny edit). This is the failure the fact-token check exists for.

## What this does and doesn't show
- Shows: Stage 1 never blocks on semantics, and its known false drops occur on the number-form cases above; a string baseline is no better than the label prior here.
- Doesn't show: any Stage 2 performance, any real-world precision, any significance.
