# Verifier limitations: known false drops

Stage 1 requires every number, date and capitalised mid-sentence word of the claim to appear **exactly** in the cited
span (case-insensitive, `missing_fact_tokens` in `app/core/verifier.py`). It has no alias, rounding or synonym
knowledge. A false drop costs coverage; a false keep costs precision, which is the product. The checks are therefore
**not relaxed** without measured evidence (precision/coverage on the Phase 2 held-out set). Pinned by
`tests/test_phase1_followups.py::SCENARIOS`.

| # | Case | Claim | Source span | Why it drops |
|---|------|-------|-------------|--------------|
| 1 | Rounding | "Revenue was about 12 million." | "Revenue was 12.4 million." | `12` is not the token `12.4`. (The reverse, claim 12.4 vs span "about 12", is a correct drop.) |
| 2 | Entity alias | "It was acquired by Alphabet." | "It was acquired by Google." | `alphabet` is absent from the span; no alias table. |
| 3 | Acronym expansion | "It was run by NASA staff." | "...National Aeronautics and Space Administration staff." | `nasa` is absent; expansions are not matched. |
| 4 | Capitalised non-entity | "Results were good, However tiny." | "Results were good and tiny." | Any capitalised word not at a sentence start is treated as an entity. |

Related heuristics: sentence-initial words are skipped, so a changed *first* word is not caught by Stage 1 (Stage 2
may catch it); capitalised-word checks use a lowercase word set, so "Study" matches "study".

Not covered at all by Stage 1: negation, hedging, causal overreach. Those rely on Stage 2 (an LLM), whose measured
accuracy is a Phase 2 deliverable, not a claim made here.
