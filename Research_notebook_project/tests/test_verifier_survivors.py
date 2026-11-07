"""Tests written from the surviving mutants of the baseline mutmut run on app/core/verifier.py (see TESTING.md).
Every expected value is derived by hand or from an independent construction in the test, never copied from the code's output.
Message prose (the English text of drop reasons / prompts) is intentionally NOT asserted: `drop_reason` codes are the contract."""
import hashlib
import hmac
import re
from unittest.mock import AsyncMock, patch

import pytest

from app.core.verifier import (
    MAX_SPAN_CHARS,
    _clusters,
    _fence,
    build_entailment_messages,
    check_quote_in_snapshot,
    entities_in,
    locate_quote,
    normalize_text,
    normalize_with_map,
    valid_ref,
    verify_claim,
)

# ---------- empty / degenerate input ----------

@pytest.mark.parametrize("quote", ["", "   ", "\n\t"])
def test_empty_quote_is_never_found_in_a_nonempty_snapshot(quote):
    # str.find("") == 0, so an empty quote would "match" at offset 0 if the emptiness guard used `and` instead of `or`
    assert locate_quote(quote, "Some real snapshot text.") is None
    assert check_quote_in_snapshot(quote, "Some real snapshot text.") == (False, 0.0)


def test_nonempty_quote_in_empty_snapshot_is_not_found():
    assert locate_quote("some quote", "") is None
    assert check_quote_in_snapshot("some quote", "") == (False, 0.0)


def test_normalize_helpers_return_empty_for_empty_or_none_not_a_placeholder():
    assert normalize_text("") == "" and normalize_text(None) == ""
    assert normalize_with_map("") == ("", [], [])
    assert normalize_with_map(None) == ("", [], [])


@pytest.mark.asyncio
@pytest.mark.parametrize("snap", [None, ""])
async def test_missing_snapshot_does_not_match_a_quote_that_equals_a_placeholder(snap):
    with patch("app.core.verifier.call_llm", AsyncMock()) as m:
        r = await verify_claim(None, 1, "claim", "XXXX", snap)
    assert r.drop_reason == "quote_not_found" and r.action == "dropped" and r.quote_found is False
    m.assert_not_awaited()


# ---------- check_quote_in_snapshot: not-found path ----------

def test_short_unfound_quote_scores_exactly_zero_even_if_it_shares_words_with_the_snapshot():
    # 9 chars < 40: the fuzzy scorer must not run, so the score is 0.0 although "lab" occurs in the snapshot
    snap = "The Acme lab reported a result that nobody could reproduce in any other lab."
    assert check_quote_in_snapshot("Zebra lab", snap) == (False, 0.0)


def test_unfound_quote_of_exactly_40_chars_gets_a_fuzzy_score():
    snap = "The Acme lab reported a result that nobody could reproduce in any other lab."
    q = "the acme lab reported something different"  # hand-counted below
    assert len(q) == 41
    q40 = q[:40]
    assert len(normalize_text(q40)) == 40 and normalize_text(q40) not in normalize_text(snap)
    found, score = check_quote_in_snapshot(q40, snap)
    assert found is False and 0.0 < score < 95.0


def test_long_unfound_quote_returns_false_with_a_strictly_partial_score():
    snap = "The committee published its annual budget report in March after months of debate among members."
    q = "Regional airports expanded cargo capacity substantially during the recent decade of growth"
    found, score = check_quote_in_snapshot(q, snap)
    assert found is False and 0.0 < score < 95.0


# ---------- locate_quote: fuzzy gate boundaries and occurrence choice ----------

def test_first_occurrence_is_returned_when_the_quote_appears_twice():
    sentence = "the bridge opened to traffic after a long delay"
    snap = f"intro. {sentence}. middle part. {sentence}. end."
    start, end, score, exact = locate_quote(sentence, snap)
    assert start == snap.index(sentence) and end == start + len(sentence) and exact and score == 100.0


def test_fuzzy_hit_at_exactly_40_chars_and_exactly_95_percent_is_accepted():
    # 40-char quote vs a snapshot equal except for 2 substituted letters: indel distance 4 over 80 total => ratio 95.0 by hand
    quote = "the harbor authority approved the plan.."  # 40 chars
    snap = "the harbor authority approved the plan.."
    mutated = list(snap)
    mutated[4], mutated[30] = "x", "z"  # two substitutions, letters only: no digit/entity tokens involved
    snap = "".join(mutated)
    assert len(quote) == 40 and quote != snap
    s, e, score, exact = locate_quote(quote, snap)
    assert exact is False and score == 95.0 and (s, e) == (0, 40)
    assert check_quote_in_snapshot(quote, snap) == (True, 95.0)


def test_fuzzy_hit_below_95_percent_is_rejected():
    quote = "the harbor authority approved the plan.."
    mutated = list(quote)
    mutated[4], mutated[15], mutated[30] = "x", "q", "z"  # three substitutions: 6/80 => 92.5
    assert locate_quote(quote, "".join(mutated)) is None


def test_fuzzy_hit_keeps_a_true_result_when_facts_are_present_in_the_matched_span():
    snap = "Analysts said the comittee reported that Nvidia revenue grew to 5.2 billion dollars in the fiscal year."
    quote = "the committee reported that Nvidia revenue grew to 5.2 billion dollars in the fiscal year"  # typo only
    found, score = check_quote_in_snapshot(quote, snap)
    assert found is True and 95.0 <= score < 100.0


def test_fuzzy_hit_with_a_changed_digit_is_rejected():
    snap = "Analysts said the comittee reported that Nvidia revenue grew to 5.2 billion dollars in the fiscal year."
    quote = "the committee reported that Nvidia revenue grew to 6.2 billion dollars in the fiscal year"
    found, _ = check_quote_in_snapshot(quote, snap)
    assert found is False


# ---------- valid_ref bounds ----------

def test_valid_ref_span_length_bounds_use_length_not_sum():
    snap = "a" * 5000
    assert valid_ref(snap, 0, MAX_SPAN_CHARS) is True            # exactly the cap
    assert valid_ref(snap, 0, MAX_SPAN_CHARS + 1) is False       # one over
    assert valid_ref(snap, 3000, 4999) is True                   # short span far from the start: end+start is large
    assert valid_ref(snap, 0, 5000) is False                     # whole 5000-char snapshot exceeds the cap
    assert valid_ref(snap, 4999, 5001) is False                  # past the end
    assert valid_ref(snap, 5, 5) is False and valid_ref(snap, -1, 5) is False


# ---------- entity heuristic: sentence-initial words are not entities ----------

def test_sentence_initial_capitals_are_not_entities_but_mid_sentence_ones_are():
    assert entities_in("Rates rose. Sales fell in Texas") == {"texas"}


@pytest.mark.parametrize("p", list(".!?:;\"'("))
def test_each_sentence_boundary_character_suppresses_the_next_capital(p):
    assert entities_in(f"it was said{p} Bob went to Paris") == {"paris"}


def test_multiple_spaces_after_a_boundary_still_count_as_sentence_initial():
    assert entities_in("It ended.    Maria left for Oslo") == {"oslo"}


# ---------- normalization map ----------

def test_clusters_split_base_chars_and_attach_combining_marks():
    assert list(_clusters("ab")) == [(0, 1, "a"), (1, 2, "b")]
    assert list(_clusters("é̂x")) == [(0, 3, "é̂"), (3, 4, "x")]
    assert list(_clusters("éx")) == [(0, 2, "é"), (2, 3, "x")]  # exactly one mark: must not swallow the next char
    assert list(_clusters("")) == []


def test_whitespace_run_maps_to_one_space_whose_end_covers_the_whole_run():
    # "a   b": a=[0,1) space-run=[1,4) b=[4,5) by hand
    assert normalize_with_map("a   b") == ("a b", [0, 1, 4], [1, 4, 5])
    # longer prefix so the run-extension must update the LAST end, not a fixed index: "abc" + 3 spaces + "d"
    assert normalize_with_map("abc   d") == ("abc d", [0, 1, 2, 3, 6], [1, 2, 3, 6, 7])


# ---------- Stage 2 prompt construction ----------

CLAIM, SPAN, CTX = "Acme grew.", "Acme grew 5%.", "Context before. Acme grew 5%. Context after."


def _fences(msgs):
    user = msgs[1]["content"]
    return re.findall(r"(DATA-[0-9a-f]{24})", user), user


def test_message_roles_and_shape():
    msgs = build_entailment_messages(CLAIM, SPAN, CTX)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert all(set(m) == {"role", "content"} for m in msgs)


def test_every_block_uses_one_fence_and_carries_its_own_text_in_its_own_block():
    fences, user = _fences(build_entailment_messages(CLAIM, SPAN, CTX))
    assert len(fences) == 6 and len(set(fences)) == 1  # 3 opening + 3 closing markers
    f = fences[0]
    for label, text in (("CLAIM", CLAIM), ("SOURCE SPAN", SPAN), ("SURROUNDING CONTEXT", CTX)):
        assert f"<<<{f} {label}>>>\n{text}\n<<<END {f}>>>" in user


@pytest.mark.parametrize("claim,span,ctx", [
    ("Acme shrank.", SPAN, CTX),
    (CLAIM, "Acme grew 6%.", CTX),
    (CLAIM, SPAN, CTX + " extra"),
])
def test_fence_depends_on_all_three_parts(claim, span, ctx):
    base = _fences(build_entailment_messages(CLAIM, SPAN, CTX))[0][0]
    assert _fences(build_entailment_messages(claim, span, ctx))[0][0] != base


def test_fence_matches_the_documented_hmac_construction(monkeypatch):
    # Independent re-computation of the documented scheme: HMAC-SHA256(key, concat(8-byte big-endian length || bytes)) first 24 hex.
    monkeypatch.setenv("CITEGUARD_EVIDENCE_HMAC_KEY", "k-for-test")
    parts = ("alpha", "βeta", "")
    msg = b"".join(len(p.encode()).to_bytes(8, "big") + p.encode() for p in parts)
    expected = "DATA-" + hmac.new(b"k-for-test", msg, hashlib.sha256).hexdigest()[:24]
    assert _fence(*parts) == expected


def test_fence_is_ambiguity_free_across_part_boundaries():
    # ("ab","c") and ("a","bc") must not collide: this is what the length prefix is for
    assert _fence("ab", "c") != _fence("a", "bc")
