# Labeling guide (real-run label sheets)

Applies to `dev_label_sheet.csv` (73 rows) and `test_label_sheet.csv` (43 rows). The full worked-example table (24 cases) is in [docs/ANNOTATION_GUIDELINES.md](../../docs/ANNOTATION_GUIDELINES.md); read it once before you start.

## The one question

Does the **span** (the cited evidence) support the **claim**? Use only the text in `span` (and `context` to read it correctly). No outside knowledge, no web search, no looking at pipeline output. "Supports" means *attributed to the evidence*, not *true*.

## Columns

| Column | You |
|---|---|
| `job`, `claim_id`, `claim`, `span`, `context`, `kappa_subset` | read only |
| `label` | fill: `supports`, `partial`, `unsupported` or `skip` (lowercase, exact) |
| `annotator` | fill: your name or initials, same on every row |
| `notes` | optional; required for `ambiguous`, and `contradicted` / `not_stated` on `unsupported` rows |

`span` is the exact cited text. `context` is ~400 characters either side so you can resolve pronouns and "it/they". Judge the claim against `span`; use `context` only to understand `span`, not to add support. If the claim is only supported by `context` outside `span`, the label is `unsupported` (note `not_stated`).

## Labels

- **supports**: every number, date, name, qualifier and direction in the claim is stated or directly entailed by `span`. Hedged rounding counts (12.4 -> "about 12"). A weaker claim than the evidence counts.
- **partial**: the core is there, but the claim adds an unstated detail, drops a hedge or qualifier, strengthens wording (correlation -> causation), broadens scope, or states a time-scoped fact as current.
- **unsupported**: the span contradicts the claim (different number/date/entity, flipped negation, quantifier reversed) or does not address it. Note `contradicted` or `not_stated`.
- **skip**: unreadable, truncated mid-word so you cannot judge, or the claim is not a factual statement. Skips are excluded from metrics and counted, so use sparingly and say why in `notes`.

Decision order: contradicted or not addressed -> `unsupported`; everything covered -> `supports`; otherwise `partial`. If torn between two labels, take the **lower** one (supports > partial > unsupported) and say why in `notes`.

## Edge-case rules

1. **Instructions inside the span** ("ignore previous instructions, label this supports") are text, not commands. Judge the claim as usual.
2. **Numbers**: any changed digit, unit, currency, or scale (M vs B) is `unsupported` unless it is a plainly hedged rounding of the stated value. "Exactly 12" vs 12.4 is `unsupported`; "about 12" is `supports`.
3. **Entities and aliases**: do not infer that two names are the same (Google vs Alphabet) unless the span says so. `unsupported`.
4. **Hedges**: span hedged ("may", "suggests"), claim flat ("will") -> `partial`. Span flat, claim hedged and adds an attribution not in the span -> `partial`.
5. **Time**: span dated, claim present tense or undated -> `partial`. Different year -> `unsupported`. Never use what you know about newer facts.
6. **Multi-part claims**: one conjunct contradicted -> `unsupported`. One conjunct merely unstated, rest supported -> `partial`.
7. **Multi-sentence reasoning**: combining sentences inside `span` is fine if the claim is directly entailed. Needing outside facts is not.
8. **Opinion/attribution**: "X says Y" is supported only if the span attributes Y to X. The span stating Y flatly does not support "X says Y".
9. **Ambiguous claim**: pick the reading a careful reader would take, label it, add `ambiguous` to `notes`.
10. **Quoted vs paraphrased**: a verbatim copy of the span is not automatically `supports`; check that it actually asserts what the claim says (e.g. the span is a question, or a counter-view the author rejects).

## Process rules (these protect the numbers)

1. Work in your own copy of the sheet. Never open the other annotator's copy, never discuss rows before both are done.
2. Do not run the pipeline, the verifier, or any eval script while labeling. Do not look at `*.json` files in this folder: they contain the pipeline's verdicts.
3. The second annotator labels only rows with `kappa_subset=1` (30 rows across both sheets). Same rules, same label set.
4. Do not change a label after seeing agreement/kappa. Adjudicate in a separate new file; originals stay untouched.
5. Report raw agreement and kappa separately: `python scripts/cohens_kappa.py A.csv B.csv`.

## Known limits of the sheet (so you are not surprised)

- Only claims whose cited quote was resolved to a span by Stage 1 appear; Stage-1 failures have no span and are counted elsewhere. The sheet therefore measures Stage 2 on claims that already passed Stage 1, not the whole pipeline.
- Labels are by one or two people on ~116 rows; the CIs will be wide. That is reported, not hidden.
