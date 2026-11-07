# Verifier limitations: known false drops

Stage 1 requires every number, date and capitalised mid-sentence word of the claim to appear **exactly** in the cited
span (case-insensitive, `missing_fact_tokens` in `app/core/verifier.py`). It has no alias, rounding or synonym
knowledge. A false drop costs coverage; a false keep costs precision, which is the product. The checks are therefore
**not relaxed** without measured evidence (precision/coverage on a human-labeled held-out set). Pinned by
`tests/test_phase1_followups.py::SCENARIOS`.

| # | Case | Claim | Source span | Why it drops |
|---|------|-------|-------------|--------------|
| 1 | Unhedged rounding | "Revenue was 12 million." | "Revenue was 12.4 million." | `12` is not the token `12.4`. A **hedged** claim ("about/approximately/around/roughly/nearly/almost/~ 12") now passes when a span number rounds to it at the claim's precision and unit. That relaxation follows the project spec plus one synthetic case (hard case 7), not a held-out measurement. Bounds ("just over 1,000" vs 1,046) and order-of-magnitude rounding ("about 1,000" vs 1,042) still drop. |
| 1b | Units | "It weighed 12 kg." | "It weighed 12 lb." | Not a drop: only % and scale words (hundred, thousand, million, billion, trillion, lakh, crore) are compared, so other units pass Stage 1 and rely on Stage 2. |
| 2 | Entity alias | "It was acquired by Alphabet." | "It was acquired by Google." | `alphabet` is absent from the span; no alias table. |
| 3 | Acronym expansion | "It was run by NASA staff." | "...National Aeronautics and Space Administration staff." | `nasa` is absent; expansions are not matched. |
| 4 | Capitalised non-entity | "Results were good, However tiny." | "Results were good and tiny." | Any capitalised word not at a sentence start is treated as an entity. |

Related heuristics: sentence-initial words are skipped, so a changed *first* word is not caught by Stage 1 (Stage 2
may catch it); capitalised-word checks use a lowercase word set, so "Study" matches "study".

Not covered at all by Stage 1: negation, hedging, causal overreach. Those rely on Stage 2 (an LLM), whose measured
accuracy on real claims is not measured yet, so no claim is made here.
