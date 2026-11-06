"""Hypothesis property tests. Expected values come from how the inputs are CONSTRUCTED, never from the function under test."""
import re

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from app.core.fetcher import chunk_spans
from app.core.verifier import locate_quote, normalize_text, normalize_with_map

# words with letters/digits/punctuation plus awkward unicode (curly quotes, NBSP, combining marks, fullwidth)
WORD = st.text(alphabet=st.characters(categories=("L", "N", "P", "Zs", "Mn"), max_codepoint=0x2FFF), min_size=1, max_size=12)
TEXT = st.lists(WORD, min_size=1, max_size=40).map(" ".join)
PLAIN = st.lists(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=3, max_size=9), min_size=25, max_size=40).map(" ".join)


@given(TEXT)
def test_normalization_is_idempotent(t):
    once = normalize_text(t)
    assert normalize_text(once) == once


@given(TEXT)
def test_normalization_map_is_well_formed(t):
    n, starts, ends = normalize_with_map(t)
    assert len(n) == len(starts) == len(ends)
    assert all(0 <= s < e <= len(t) for s, e in zip(starts, ends))
    assert starts == sorted(starts)


@settings(max_examples=200)
@given(snap=TEXT, data=st.data())
def test_quote_copied_from_snapshot_always_found_and_maps_back(snap, data):
    n = normalize_text(snap)
    if len(n) < 3:
        return
    i = data.draw(st.integers(0, len(n) - 2))
    j = data.draw(st.integers(i + 1, len(n)))
    quote = n[i:j].strip()
    if not quote:
        return
    loc = locate_quote(quote, snap)
    assert loc is not None, (snap, quote)
    s, e, score, exact = loc
    assert exact and score == 100.0
    assert quote in normalize_text(snap[s:e])  # the raw span really contains the quote


@settings(max_examples=100)
@given(prefix=PLAIN, suffix=PLAIN, num=st.integers(10, 99999), pos=st.integers(0, 4), delta=st.integers(1, 9))
def test_any_changed_digit_in_a_long_quote_is_found_by_fuzzy_but_rejected_by_guard(prefix, suffix, num, pos, delta):
    from app.core.verifier import check_quote_in_snapshot, missing_fact_tokens
    digits = str(num)
    pos %= len(digits)
    bad_digit = str((int(digits[pos]) + delta) % 10)
    bad = digits[:pos] + bad_digit + digits[pos + 1:]
    assert bad != digits
    snap = f"{prefix} measured {digits} units {suffix}"
    quote_bad = f"{prefix} measured {bad} units {suffix}"
    loc = locate_quote(quote_bad, snap)
    # long quote: one wrong digit stays above the fuzzy threshold, so the fact-token guard is what must reject it
    assert loc is not None and not loc[3], "precondition: fuzzy alone would have accepted this"
    assert missing_fact_tokens(quote_bad, snap[loc[0]:loc[1]])
    assert check_quote_in_snapshot(quote_bad, snap) == (False, loc[2])


@given(TEXT.map(lambda t: t + " ").flatmap(lambda t: st.just(t * 30)), st.integers(5, 40), st.integers(0, 4))
def test_chunk_spans_slice_back_to_exact_text(t, size, overlap):
    assume(t.strip())  # whitespace-only text has no words, so correctly no chunks
    spans = chunk_spans(t, chunk_size=size, overlap=overlap)
    words = [m.group() for m in re.finditer(r"\S+", t)]
    assert spans, "non-empty text must give chunks"
    assert t[spans[0][0]:spans[0][1]].split()[0] == words[0]
    assert t[spans[-1][0]:spans[-1][1]].split()[-1] == words[-1]
    for a, b in spans:
        assert 0 <= a < b <= len(t) and not t[a].isspace() and not t[b - 1].isspace()
        assert len(t[a:b].split()) <= size


WORDS = st.lists(st.text(alphabet="abcxyz019", min_size=1, max_size=6), min_size=1, max_size=15)
WS = st.sampled_from([" ", "  ", "\t", "\n", " \r\n ", "\u00a0", "\u2003"])


@given(words=WORDS, seps=st.lists(WS, min_size=15, max_size=15), lead=WS, trail=WS)
def test_any_whitespace_runs_collapse_to_single_spaces(words, seps, lead, trail):
    text = lead + "".join(w + seps[i] for i, w in enumerate(words)) + trail
    assert normalize_text(text) == " ".join(words)  # expected built from the words, not from the function


@given(st.integers(0, 10**6), st.integers(0, 99))
def test_fullwidth_digits_fold_to_ascii_via_nfkc(whole, frac):
    ascii_num = f"{whole}.{frac}"
    fullwidth = "".join(chr(ord(c) + 0xFEE0) if c.isdigit() else c for c in ascii_num)
    assert fullwidth != ascii_num
    assert normalize_text(f"rose {fullwidth}%") == f"rose {ascii_num}%"


def test_regression_dotted_capital_i_with_combining_mark_is_idempotent_and_findable():
    """Found by Hypothesis under randomized seeds: U+0130 lowercases to i + U+0307 AFTER NFKC, leaving combining marks
    out of canonical order, so a quote copied from the normalized snapshot normalized differently the second time."""
    snap = "00000\u0130\u05ad"
    once = normalize_text(snap)
    assert normalize_text(once) == once
    assert locate_quote(once, snap) is not None
