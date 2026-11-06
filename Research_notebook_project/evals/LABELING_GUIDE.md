# Labeling guide (read before annotating)

You see a CLAIM and the SOURCE SPAN it cites (plus context). Judge only whether the span supports the claim.
Do not use outside knowledge. "Supports" means attributed to the span, not true in the world.
You do NOT see any model verdict (blind labeling).

| Label | Definition |
|---|---|
| `supports` | Every part of the claim (numbers, dates, names, qualifiers, direction) is stated or directly entailed by the span. |
| `partial` | The main point is in the span, but the claim adds an unproven detail, drops a qualifier, or overstates slightly. |
| `unsupported` | The span does not state it, contradicts it, or the claim changes a number, date, entity, or negation. |

## Rules for hard cases
- **Rounding:** span "12.4%", claim "about 12%" -> `supports`. Claim "12%" stated as exact, or "13%" -> `unsupported`. Claim "over 15%" -> `unsupported`.
- **Negation:** span "no significant effect", claim "had a significant effect" or "had an effect" -> `unsupported`.
- **Hedging:** span "may reduce risk", claim "reduces risk" -> `partial`. Span "reduces risk", claim "may reduce risk" -> `supports`.
- **Overstatement:** span "some patients", claim "patients" / "all patients" -> `partial` (`unsupported` if it flips the meaning).
- **Causation:** span reports correlation, claim says "causes" -> `partial` or `unsupported` (use `unsupported` if the span explicitly denies causation).
- **Multi-sentence:** the claim may combine the cited sentences; the combination must still follow from them.
- **Quote present, claim wider:** if the span is quoted accurately but the claim says more than it, label by the claim, not the quote.
- **Unclear / span cut off / claim is not factual:** use `skip` and write a note. Skips are reported, not hidden.

## Examples
1. Span: "The cell reached 33.9% efficiency in 2023." Claim: "The cell reached about 34% efficiency in 2023." -> supports
2. Span: "The cell reached 33.9% efficiency in 2023." Claim: "The cell reached 34.9% efficiency." -> unsupported (number)
3. Span: "Remote workers may be more productive." Claim: "Remote workers are more productive." -> partial (hedge dropped)
4. Span: "The study found no link between X and Y." Claim: "X is linked to Y." -> unsupported (negation)

Annotator 1 labels everything. Annotator 2 independently labels the rows marked `kappa_subset=1` without seeing annotator 1's labels.
