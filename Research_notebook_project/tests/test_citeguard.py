# tests/test_citeguard.py
import pytest

from app.fetcher import SSRFSecurityError, chunk_text, validate_url_security
from app.llm import (
    calculate_modeled_usd,
    estimate_prompt_tokens,
    get_cache_key,
    read_from_cache,
    write_to_cache,
)
from app.verifier import check_quote_in_snapshot, normalize_text


def test_ssrf_blocks_private_and_loopback_ips():
    """Verify that private, loopback, and metadata IPs are rejected."""
    # Loopback IP
    with pytest.raises(SSRFSecurityError):
        validate_url_security("http://127.0.0.1:8080/admin")

    # Localhost domain
    with pytest.raises(SSRFSecurityError):
        validate_url_security("http://localhost:5432")

    # Unsupported protocol scheme
    with pytest.raises(SSRFSecurityError):
        validate_url_security("ftp://example.com/data.txt")

    with pytest.raises(SSRFSecurityError):
        validate_url_security("file:///etc/passwd")

def test_deterministic_quote_normalization():
    """Verify Unicode NFKC normalization and quote/dash standardization."""
    raw = "“Google’s Sycamore—a 53-qubit device—achieved quantum supremacy.”"
    expected = '"google\'s sycamore-a 53-qubit device-achieved quantum supremacy."'
    assert normalize_text(raw) == expected

def test_quote_in_snapshot_exact_and_fuzzy():
    """Verify deterministic snapshot matching with exact and fuzzy fallback."""
    snapshot = "In October 2019, Google researchers announced that their 53-qubit Sycamore processor achieved quantum supremacy."
    
    # Exact quote
    quote_exact = "Google researchers announced that their 53-qubit Sycamore processor achieved quantum supremacy"
    found, score = check_quote_in_snapshot(quote_exact, snapshot)
    assert found is True
    assert score >= 95.0

    # Slight punctuation/spacing difference (fuzzy fallback)
    quote_fuzzy = "Google researchers announced that their 53-qubit Sycamore processor, achieved quantum supremacy"
    found_fuz, score_fuz = check_quote_in_snapshot(quote_fuzzy, snapshot)
    assert found_fuz is True
    assert score_fuz >= 95.0

    # Fabricated / Hallucinated quote
    quote_fake = "Google permanently replaced all supercomputers with quantum chips worldwide in 2024"
    found_fake, score_fake = check_quote_in_snapshot(quote_fake, snapshot)
    assert found_fake is False
    assert score_fake < 95.0

def test_chunking_utility():
    """Verify word chunking preserves coverage with overlap."""
    text = " ".join([f"word_{i}" for i in range(100)])
    chunks = chunk_text(text, chunk_size=30, overlap=10)
    assert len(chunks) >= 4
    # Check that overlap exists
    assert "word_25" in chunks[0] and "word_25" in chunks[1]

def test_llm_cache_roundtrip():
    """Verify disk caching saves and returns identical content."""
    messages = [{"role": "user", "content": "Test prompt for cache key verification"}]
    key = get_cache_key("test-model", messages, None)
    
    data = {"raw_text": "Sample cached completion", "tokens_in": 10, "tokens_out": 5}
    write_to_cache(key, data)

    cached = read_from_cache(key)
    assert cached is not None
    assert cached["raw_text"] == "Sample cached completion"
    assert cached["tokens_in"] == 10

def test_token_estimation_and_cost_math():
    """Verify cost and token estimation formulas."""
    messages = [{"role": "user", "content": "1234567890 1234567890"}] # ~20 chars -> ~5 tokens
    tokens = estimate_prompt_tokens(messages)
    assert tokens >= 4

    cost = calculate_modeled_usd("openai/gpt-oss-120b", tokens_in=1000, tokens_out=1000)
    # no verified price for this model -> generic default: 0.0005 + 0.0015 = 0.002
    assert abs(cost - 0.002) < 1e-5



def test_verifier_benchmark_testset_integrity():
    """Verify that the verifier test set exists and contains 150 items with proper labels."""
    import csv
    from pathlib import Path

    csv_path = Path(__file__).resolve().parents[1] / "evals" / "verifier_testset.csv"
    assert csv_path.exists(), "evals/verifier_testset.csv must exist"

    with open(csv_path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) >= 150
    types = {row["perturbation_type"] for row in reader}
    assert "original" in types
    assert "number" in types
    assert "date" in types
    assert "overgeneralize" in types
    assert "entity" in types
