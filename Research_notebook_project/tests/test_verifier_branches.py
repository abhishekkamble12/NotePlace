"""Branch-level tests for verify_claim and rewrite_claim, written from mutation survivors (see TESTING.md).
Providers are mocked; these pin routing, drop codes, offsets, match scores and the exact LLM call contract."""
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from rapidfuzz import fuzz

from app.core.llm import BudgetExceededError, ProviderUnavailableError
from app.core.verifier import (
    EntailmentVerdict,
    rewrite_claim,
    verify_claim,
)
from app.research import nodes

SNAP = ("Background text. In late 2023 the Acme lab reported a peak efficiency of 33.9% for the new perovskite cell, "
        "measured by Fraunhofer ISE in Freiburg. More text follows here.")
QUOTE = "In late 2023 the Acme lab reported a peak efficiency of 33.9% for the new perovskite cell, measured by Fraunhofer ISE in Freiburg."
S = SNAP.index("In late 2023")
E = S + len(QUOTE)
CLAIM = "Acme reported 33.9% efficiency."  # all facts present in the span


def verdict(v="supports"):
    return EntailmentVerdict(verdict=v, reason="r")


# ---------- routing: which branch runs ----------

@pytest.mark.asyncio
async def test_only_start_or_only_end_still_uses_reference_branch():
    # `start is not None or end is not None`: one half of the reference must be enough to skip the quote path
    for st, en in [(S, None), (None, E)]:
        with patch("app.core.verifier.call_llm", AsyncMock()) as m:
            r = await verify_claim(None, 1, CLAIM, QUOTE, SNAP, start=st, end=en)
        assert r.drop_reason == "invalid_citation_ref" and r.quote_found is False and r.match_score == 0.0
        m.assert_not_awaited()


@pytest.mark.asyncio
async def test_reference_branch_scores_exact_100_and_reports_span():
    mock = AsyncMock(return_value=verdict())
    with patch("app.core.verifier.call_llm", mock):
        r = await verify_claim(None, 1, CLAIM, "", SNAP, start=S, end=E)
    assert r.match_score == 100.0 and r.quote_found is True and (r.span_start, r.span_end) == (S, E)


@pytest.mark.asyncio
async def test_invalid_reference_result_fields():
    r = await verify_claim(None, 1, CLAIM, "", SNAP, start=-1, end=5)
    assert (r.quote_found, r.match_score, r.span_start, r.span_end) == (False, 0.0, None, None)


# ---------- Stage 1 quote path: fuzzy score and facts ----------

@pytest.mark.asyncio
async def test_quote_not_found_reports_a_real_fuzzy_score_for_40_char_quotes():
    # long enough for the fuzzy gate (>= MIN_FUZZY_QUOTE_CHARS): the score must be computed, not zeroed
    q = "Acme lab reported 99 percent of peaks!!!!!!!!"  # 46 chars, not in snapshot
    r = await verify_claim(None, 1, CLAIM, q, SNAP)
    assert r.drop_reason == "quote_not_found" and r.quote_found is False
    expected = float(fuzz.partial_ratio(q.lower(), SNAP.lower()))
    assert r.match_score == pytest.approx(expected, abs=1.0) and r.match_score < 95.0


@pytest.mark.asyncio
async def test_short_quote_gets_zero_score_not_fuzzy_score():
    r = await verify_claim(None, 1, CLAIM, "Zenith lab", SNAP)
    assert r.drop_reason == "quote_not_found" and r.match_score == 0.0


@pytest.mark.asyncio
async def test_empty_snapshot_gets_zero_score():
    r = await verify_claim(None, 1, CLAIM, QUOTE, "")
    assert r.drop_reason == "quote_not_found" and r.match_score == 0.0 and r.quote_found is False


@pytest.mark.asyncio
async def test_fuzzy_accepted_quote_facts_checked_against_span_not_just_claim():
    # Claim has no numbers, but the writer's quote changed 33.9% -> 43.9%; fuzzy alone would accept it.
    bad_quote = QUOTE.replace("33.9%", "43.9%")
    with patch("app.core.verifier.call_llm", AsyncMock()) as m:
        r = await verify_claim(None, 1, "Acme lab reported strong efficiency.", bad_quote, SNAP)
    assert r.drop_reason == "fact_token_mismatch" and r.quote_found is True
    assert (r.span_start, r.span_end) == (S, E) and r.match_score >= 95.0
    m.assert_not_awaited()


@pytest.mark.asyncio
async def test_fact_mismatch_reports_span_and_found_flag_on_reference_path():
    with patch("app.core.verifier.call_llm", AsyncMock()) as m:
        r = await verify_claim(None, 1, "The cell reached 43.9% efficiency.", "", SNAP, start=S, end=E)
    assert r.drop_reason == "fact_token_mismatch" and r.quote_found is True
    assert (r.span_start, r.span_end) == (S, E) and r.match_score == 100.0
    m.assert_not_awaited()


# ---------- Stage 2: call contract ----------

@pytest.mark.asyncio
async def test_llm_call_contract_is_exact():
    mock = AsyncMock(return_value=verdict())
    with patch("app.core.verifier.call_llm", mock):
        await verify_claim(None, 42, CLAIM, "", SNAP, start=S, end=E)
    kw = mock.call_args.kwargs
    assert kw["scope"] == ("job", 42)
    assert kw["phase"] == "verifying"
    assert kw["model"] == "qwen/qwen3.8-27b"
    assert kw["max_tokens"] == 250
    assert kw["schema"] is EntailmentVerdict
    assert kw["db"] is None and kw["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_custom_verifier_model_is_passed_through():
    mock = AsyncMock(return_value=verdict())
    with patch("app.core.verifier.call_llm", mock):
        await verify_claim(None, 1, CLAIM, "", SNAP, verifier_model="other/model", start=S, end=E)
    assert mock.call_args.kwargs["model"] == "other/model"


@pytest.mark.asyncio
async def test_context_window_surrounds_the_span():
    mock = AsyncMock(return_value=verdict())
    with patch("app.core.verifier.call_llm", mock):
        await verify_claim(None, 1, CLAIM, "", SNAP, start=S, end=E)
    user = mock.call_args.kwargs["messages"][1]["content"]
    # the span is 16..~130 chars into a 230-char snapshot, so both sides fall inside the 400-char window
    assert "Background text." in user and "More text follows here." in user


# ---------- verdict -> action mapping (nodes.py routes on these strings) ----------

@pytest.mark.asyncio
@pytest.mark.parametrize("v,action", [("supports", "kept"), ("partial", "needs_rewrite"), ("unsupported", "dropped")])
async def test_verdict_maps_to_action(v, action):
    with patch("app.core.verifier.call_llm", AsyncMock(return_value=verdict(v))):
        r = await verify_claim(None, 1, CLAIM, "", SNAP, start=S, end=E)
    assert r.action == action and r.verdict == v and r.reason == "r"
    assert r.quote_found is True and r.drop_reason == ""


# ---------- every drop code, with its found flag and span ----------

@pytest.mark.asyncio
async def test_invalid_structured_output_drop_codes():
    bad = None
    try:
        EntailmentVerdict(verdict="bogus", reason="x")
    except ValidationError as e:
        bad = e
    for effect, ret in [(bad, None), (None, None)]:
        mock = AsyncMock(side_effect=effect, return_value=ret)
        with patch("app.core.verifier.call_llm", mock):
            r = await verify_claim(None, 1, CLAIM, "", SNAP, start=S, end=E)
        assert r.drop_reason == "verifier_invalid_output" and r.action == "dropped"
        assert r.quote_found is True and (r.span_start, r.span_end) == (S, E)


@pytest.mark.asyncio
@pytest.mark.parametrize("exc,code", [
    (ProviderUnavailableError("down"), "provider_unavailable"),
    (BudgetExceededError("cap"), "budget_exceeded"),
])
async def test_provider_and_budget_drops_keep_found_flag_and_span(exc, code):
    with patch("app.core.verifier.call_llm", AsyncMock(side_effect=exc)):
        r = await verify_claim(None, 1, CLAIM, "", SNAP, start=S, end=E)
    assert r.drop_reason == code and r.action == "dropped" and r.verdict == "unsupported"
    assert r.quote_found is True and (r.span_start, r.span_end) == (S, E)


@pytest.mark.asyncio
async def test_missing_evidence_key_drops_as_unconfigured(monkeypatch):
    monkeypatch.delenv("CITEGUARD_EVIDENCE_HMAC_KEY", raising=False)
    mock = AsyncMock()
    with patch("app.core.verifier.call_llm", mock):
        r = await verify_claim(None, 1, CLAIM, "", SNAP, start=S, end=E)
    assert r.drop_reason == "verifier_unconfigured" and r.action == "dropped"
    assert r.quote_found is True and (r.span_start, r.span_end) == (S, E)
    mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_quote_not_found_drop_code_and_found_flag():
    r = await verify_claim(None, 1, CLAIM, "completely unrelated sentence about cooking pasta dishes", SNAP)
    assert r.drop_reason == "quote_not_found" and r.quote_found is False and r.span_start is None


# ---------- rewrite_claim: fail-closed contract ----------

@pytest.mark.asyncio
async def test_rewrite_returns_text_and_fences_the_snapshot_span():
    mock = AsyncMock(return_value=SimpleNamespace(rewritten_claim="Acme reported 33.9%."))
    with patch("app.core.verifier.call_llm", mock):
        out = await rewrite_claim(None, 7, "orig", QUOTE, "too broad")
    assert out == "Acme reported 33.9%."
    kw = mock.call_args.kwargs
    assert kw["scope"] == ("job", 7) and kw["phase"] == "verifying" and kw["max_tokens"] == 300
    assert kw["model"] == "openai/gpt-oss-120b"
    user = kw["messages"][1]["content"]
    fence = re.search(r"<<<(DATA-[0-9a-f]+) SOURCE SPAN>>>", user).group(1)
    assert f"<<<{fence} SOURCE SPAN>>>\n{QUOTE}\n<<<END {fence}>>>" in user


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", [ProviderUnavailableError("down"), BudgetExceededError("cap"),
                                 ValidationError.from_exception_data("x", [])])
async def test_rewrite_fails_closed_on_provider_budget_or_invalid_output(exc):
    with patch("app.core.verifier.call_llm", AsyncMock(side_effect=exc)):
        assert await rewrite_claim(None, 1, "orig", QUOTE, "critique") is None


@pytest.mark.asyncio
async def test_rewrite_returns_none_when_llm_gives_no_result():
    with patch("app.core.verifier.call_llm", AsyncMock(return_value=None)):
        assert await rewrite_claim(None, 1, "orig", QUOTE, "critique") is None


@pytest.mark.asyncio
async def test_rewrite_without_evidence_key_makes_no_call(monkeypatch):
    monkeypatch.delenv("CITEGUARD_EVIDENCE_HMAC_KEY", raising=False)
    mock = AsyncMock()
    with patch("app.core.verifier.call_llm", mock):
        assert await rewrite_claim(None, 1, "orig", QUOTE, "critique") is None
    mock.assert_not_awaited()


# ---------- rewrite_node: Stage 2 receives the snapshot span, never the writer's quote ----------

class _FakeDB:
    def __init__(self, objs):
        self.objs = objs
        self.committed = 0

    async def get(self, model, pk):
        return self.objs.get((model.__name__, pk))

    async def commit(self):
        self.committed += 1

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


def _run_rewrite_node(monkeypatch, claim, source, rewrite_result):
    db = _FakeDB({("Claim", 1): claim, ("Source", claim.source_id): source})
    monkeypatch.setattr(nodes, "AsyncSessionLocal", lambda: db)
    rc = AsyncMock(return_value=rewrite_result)
    monkeypatch.setattr(nodes, "rewrite_claim", rc)
    state = {"job_id": 5, "claims": [{"id": 1, "final_status": "needs_rewrite", "rewrite_count": 0}]}
    return rc, db, state


@pytest.mark.asyncio
async def test_rewrite_node_sends_snapshot_span_not_writer_quote(monkeypatch):
    forged = "Acme proved cold fusion works at 99.9% efficiency."
    claim = SimpleNamespace(id=1, text=CLAIM, quote=forged, source_id=9, start_char=S, end_char=E,
                            subq_id=1, quote_found=True, entailment="partial", entailment_reason="too broad", final_status="needs_rewrite",
                            rewrite_count=0, confidence=None, shadow_verdict=None)
    rc, db, state = _run_rewrite_node(monkeypatch, claim, SimpleNamespace(snapshot=SNAP), "Acme reported 33.9%.")
    await nodes.rewrite_node(state)
    assert rc.call_args.kwargs["quote"] == QUOTE and "cold fusion" not in rc.call_args.kwargs["quote"]
    assert claim.text == "Acme reported 33.9%." and claim.final_status == "pending" and claim.rewrite_count == 1


@pytest.mark.asyncio
async def test_rewrite_node_drops_claim_without_offsets_instead_of_using_writer_text(monkeypatch):
    claim = SimpleNamespace(id=1, text=CLAIM, quote=QUOTE, source_id=9, start_char=None, end_char=None,
                            subq_id=1, quote_found=True, entailment="partial", entailment_reason="x", final_status="needs_rewrite",
                            rewrite_count=0, confidence=None, shadow_verdict=None)
    rc, db, state = _run_rewrite_node(monkeypatch, claim, SimpleNamespace(snapshot=SNAP), "never used")
    await nodes.rewrite_node(state)
    rc.assert_not_awaited()
    assert claim.final_status == "dropped"


@pytest.mark.asyncio
async def test_rewrite_node_drops_claim_when_rewrite_fails_closed(monkeypatch):
    claim = SimpleNamespace(id=1, text=CLAIM, quote=QUOTE, source_id=9, start_char=S, end_char=E,
                            subq_id=1, quote_found=True, entailment="partial", entailment_reason="x", final_status="needs_rewrite",
                            rewrite_count=0, confidence=None, shadow_verdict=None)
    rc, db, state = _run_rewrite_node(monkeypatch, claim, SimpleNamespace(snapshot=SNAP), None)
    await nodes.rewrite_node(state)
    assert claim.final_status == "dropped" and claim.rewrite_count == 0
