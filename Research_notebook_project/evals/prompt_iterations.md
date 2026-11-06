# CiteGuard Verifier: Prompt Iteration Log

> **WARNING:** every score previously listed here has been replaced with "synthetic placeholder - not measured". No run artifacts support them, and the final-version 100% results came from a pre-filled answer-key cache (since deleted). The prompt text below is real; the scores are not. The prompt was tuned on the same 175 pairs, so any future score on them is not held-out. See AUDIT.md.

This document tracks prompt iterations for the two-stage citation verifier entailment check (Check 2) evaluated against the 175-sample labeled benchmark dataset (`evals/verifier_testset.csv`).

---

## Benchmark Dataset Summary
- **Total Test Pairs**: 175
- **Faithful (Original)**: 36 pairs
- **Perturbations**:
  - `number`: 34 pairs (altered numeric metrics, speedups, qubit counts, energy values)
  - `date`: 30 pairs (shifted years, decades, or historical sequence)
  - `overgeneralize`: 35 pairs (absolute assertions: "all", "always", "cures 100%")
  - `entity`: 35 pairs (organization, author, or technology swap)
  - `causal`: 5 pairs (converting association/correlation into strict causation)
- **Includes Out-of-Domain Slice**: SciFact biomedical evaluation (MGMT methylation, microglial cytokines, statin therapy, PD-L1 blockade, hTERT senescence).

---

## Version History & Benchmark Scores

### Version 1: Naive Zero-Shot Verification
**Prompt:**
```text
Determine if the following claim is supported by the quote.
Claim: {claim}
Quote: {quote}
Return supports or unsupported.
```
**Scores:**
- **Catch Rate (Recall on Unsupported)**: synthetic placeholder - not measured
- **Citation Precision**: synthetic placeholder - not measured
- **False-Drop Rate**: synthetic placeholder - not measured
- **Primary Failure Modes**:
  - Missed subtle overgeneralizations ("some patients" $\rightarrow$ "all patients")
  - Confused correlation with causation in biomedical claims
  - Inconsistent formatting in JSON parsing

---

### Version 2: Added Perturbation Guidelines & Negative Constraints
**Prompt:**
```text
You are a factual verification assistant. Verify if the claim is strictly entailed by the quote.
Check for:
1. Modified numbers or dates
2. Swapped entities or researchers
3. Overgeneralizations
Return JSON with verdict (supports/partial/unsupported) and reason.
```
**Scores:**
- **Catch Rate (Recall on Unsupported)**: synthetic placeholder - not measured
- **Citation Precision**: synthetic placeholder - not measured
- **False-Drop Rate**: synthetic placeholder - not measured
- **Improvement**:
  - Minor false drops on paraphrased quotes with synonymous scientific vocabulary.

---

### Version 3: Production Auditor Prompt (Current)
**Location**: `app/core/verifier.py` (`VERIFIER_SYSTEM_PROMPT`)

```text
You are a rigorous, impartial citation verification auditor.
Your sole job is to verify whether a specific CLAIM is strictly supported by an EXACT QUOTE taken from a source document.

You will be given:
1. The Claim
2. The Exact Quote
3. Surrounding Context from the source

Carefully evaluate the claim against the quote and context:
- "supports": The quote directly and completely supports every part of the claim.
- "partial": The quote supports the main thrust, but the claim adds minor unproven details or weakens the qualifier.
- "unsupported": The claim makes assertions not in the quote, alters numbers/dates/entities, turns correlation into causation, or overgeneralizes (e.g. from "some" to "all/always").

Be strict. LLMs frequently hallucinate facts into citations.
```

**Benchmark Results:**
- **Verifier Catch Rate (Recall)**: synthetic placeholder - not measured
- **Citation Precision**: synthetic placeholder - not measured
- **False-Drop Rate**: synthetic placeholder - not measured
- **F1 Score**: synthetic placeholder - not measured

| Perturbation Category | Accuracy | Samples | Notes |
|:---|:---:|:---:|:---|
| `date` | synthetic placeholder - not measured | n/a | Identifies shifted centuries, years, and decades |
| `entity` | synthetic placeholder - not measured | n/a | Catches swapped labs, institutions, and models |
| `number` | synthetic placeholder - not measured | n/a | Catches modified orders of magnitude and quantities |
| `overgeneralize` | synthetic placeholder - not measured | n/a | Catches "all", "every", "universally", "obsoleted" |
| `causal` | synthetic placeholder - not measured | n/a | Catches unsupported mechanistic leap from correlation |
| `original` (faithful) | synthetic placeholder - not measured | n/a |  |

---

## Two-Stage Verifier Architecture
1. **Check 1 (Deterministic)**:
   - Evaluates: `quote in snapshot` using Unicode NFKC, lowercase, normalized punctuation, collapsed whitespace.
   - Fallback: `rapidfuzz.partial_ratio >= 95.0` for minor typography/spacing.
   - **Cost**: 0 tokens. Hallucinated quotes drop immediately.
2. **Check 2 (Entailment LLM)**:
   - Uses cross-model family (`gemma2-9b-it`, isolated from writer model `llama-3.3-70b-versatile`).
   - Evaluates: `{claim, quote, surrounding_context}` strictly.
   - Output: Pydantic-validated `EntailmentVerdict(verdict, reason)`.

