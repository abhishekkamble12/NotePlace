"""Align a QASPER evidence paragraph (from the dataset's LaTeX-derived full text) to character offsets in OUR PDF snapshot.

Gold offsets are therefore found by matching, never invented: an evidence paragraph that cannot be located with a high
score is dropped and counted. Matching runs on alphanumeric-only, NFKC-lowercased text with an offset map back to the
snapshot, so hyphenation, line breaks, ligatures and spacing differences between the two text sources do not matter.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

MIN_SCORE = 90.0          # rapidfuzz partial_ratio on normalized text; below this the evidence is dropped
LEN_RATIO = (0.75, 1.35)  # aligned snapshot length / evidence length must be plausible, else the alignment is degenerate
MIN_EVIDENCE_CHARS = 60   # shorter normalized evidence matches almost anywhere
_DATASET_TOKENS = re.compile(r"(?:BIBREF|INLINEFORM|DISPLAYFORM|TABREF|FIGREF|SECREF|ALGREF|EQREF)\d*")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Alphanumeric-only lowercase NFKC text and, for each kept char, its index in the original string."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize("NFKC", ch):
            if c.isalnum():
                out.append(c.lower())
                idx.append(i)
    return "".join(out), idx


def align(evidence: str, snapshot: str, snap_norm: tuple[str, list[int]] | None = None) -> dict | None:
    """{'start','end','score','method'} with snapshot[start:end] the matched region, or None if not found."""
    ev, _ = normalize_with_map(_DATASET_TOKENS.sub(" ", evidence))
    if len(ev) < MIN_EVIDENCE_CHARS:
        return None
    sn, smap = snap_norm or normalize_with_map(snapshot)
    pos = sn.find(ev)
    if pos >= 0:
        a, b, score, method = pos, pos + len(ev), 100.0, "exact"
    else:
        al = fuzz.partial_ratio_alignment(ev, sn, score_cutoff=MIN_SCORE)
        if al is None or not (LEN_RATIO[0] <= (al.dest_end - al.dest_start) / len(ev) <= LEN_RATIO[1]):
            return None
        a, b, score, method = al.dest_start, al.dest_end, float(al.score), "fuzzy"
    return {"start": smap[a], "end": smap[b - 1] + 1, "score": round(score, 1), "method": method}
