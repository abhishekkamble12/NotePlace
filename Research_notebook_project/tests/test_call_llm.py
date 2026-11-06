# tests/test_call_llm.py
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from app.core.llm import (
    PHASE_BUDGET_SHARES,
    BudgetExceededError,
    RateLimitExceededError,
    SlidingWindowRateLimiter,
    call_llm,
    check_budget_caps,
    get_cache_key,
    write_to_cache,
)
from app.models import Job, LedgerEntry


class SampleOutput(BaseModel):
    summary: str
    item_count: int


# =====================================================================
# 1. Sliding Window Rate Limiter with Fake Clock
# =====================================================================

@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_allows_under_limit():
    simulated_time = 1000.0

    def fake_clock():
        return simulated_time

    sleep_calls = []

    async def fake_sleep(secs: float):
        nonlocal simulated_time
        sleep_calls.append(secs)
        simulated_time += secs

    limiter = SlidingWindowRateLimiter(
        tpm_limit=1000,
        clock_fn=fake_clock,
        sleep_fn=fake_sleep
    )

    # First acquisition of 600 tokens should be allowed immediately
    await limiter.acquire(estimated_tokens=600, model="test-model")
    assert len(sleep_calls) == 0


@pytest.mark.asyncio
async def test_sliding_window_rate_limiter_waits_when_limit_reached():
    simulated_time = 1000.0

    def fake_clock():
        return simulated_time

    sleep_calls = []

    async def fake_sleep(secs: float):
        nonlocal simulated_time
        sleep_calls.append(secs)
        simulated_time += secs

    limiter = SlidingWindowRateLimiter(
        tpm_limit=1000,
        clock_fn=fake_clock,
        sleep_fn=fake_sleep
    )

    # Acquire 800 tokens at t=1000
    await limiter.acquire(estimated_tokens=800, model="test-model")

    # At t=1010, attempting to acquire 300 tokens (800+300=1100 > 1000) must wait
    simulated_time = 1010.0
    await limiter.acquire(estimated_tokens=300, model="test-model")

    # Oldest record was at 1000.0, so wait must be ~ (60 - 10) = 50s
    assert len(sleep_calls) == 1
    assert sleep_calls[0] >= 50.0
    # After sleep, simulated_time should have advanced past 1060
    assert simulated_time >= 1060.0


@pytest.mark.asyncio
async def test_rate_limiter_exceeds_deadline_raises_error():
    simulated_time = 1000.0

    def fake_clock():
        return simulated_time

    limiter = SlidingWindowRateLimiter(
        tpm_limit=1000,
        clock_fn=fake_clock
    )

    # Acquire 900 tokens at t=1000
    await limiter.acquire(estimated_tokens=900, model="test-model")

    # Next call needs wait of ~50s, but deadline is only 10s away
    deadline = datetime.fromtimestamp(1010.0, tz=UTC)
    simulated_time = 1010.0

    with pytest.raises(RateLimitExceededError):
        await limiter.acquire(estimated_tokens=200, model="test-model", deadline_at=deadline)


# =====================================================================
# 2. Phase Cap Math and BudgetExceededError
# =====================================================================

@pytest.mark.asyncio
async def test_phase_budget_cap_math():
    """Verify phase shares: Research 50%, Writing 15%, Verify 25%, Reserve 10%."""
    assert PHASE_BUDGET_SHARES["research"] == 0.50
    assert PHASE_BUDGET_SHARES["writing"] == 0.15
    assert PHASE_BUDGET_SHARES["verifying"] == 0.25
    assert PHASE_BUDGET_SHARES["reserve"] == 0.10


@pytest.mark.asyncio
async def test_budget_exceeded_before_provider_call():
    """Verify check_budget_caps raises BudgetExceededError before provider is called."""
    mock_db = AsyncMock()

    # Create job with 10,000 max tokens
    job = Job(id=1, max_tokens=10_000)
    mock_db.get.return_value = job

    # Phase "writing" gets 15% -> 1,500 token cap
    # Mock ledger queries returning 1,400 tokens already used in "writing"
    mock_result = MagicMock()
    mock_row = MagicMock()
    mock_row.total_tokens = 3000
    mock_row.phase_tokens = 1400
    mock_result.one.return_value = mock_row
    mock_db.execute.return_value = mock_result

    # An estimated call of 200 tokens would push writing phase to 1,600 > 1,500
    with pytest.raises(BudgetExceededError) as exc_info:
        await check_budget_caps(mock_db, scope=("job", 1), phase="writing", estimated_tokens=200)

    assert "Phase 'writing' token cap reached" in str(exc_info.value)

    # Overall job cap exceeded test
    mock_row.total_tokens = 9500
    mock_row.phase_tokens = 500  # under writing cap, but 9500 + 600 > 10000 total cap
    with pytest.raises(BudgetExceededError) as exc_info:
        await check_budget_caps(mock_db, scope=("job", 1), phase="writing", estimated_tokens=600)

    assert "Job token budget exceeded" in str(exc_info.value)


# =====================================================================
# 3. Disk Cache Hit Costs 0 Tokens in Ledger
# =====================================================================

@pytest.mark.asyncio
async def test_cache_hit_costs_zero_tokens():
    mock_db = AsyncMock()
    mock_db.add = MagicMock()
    messages = [{"role": "user", "content": "Cache hit test question"}]
    key = get_cache_key("test-model", messages, SampleOutput, max_tokens=200)

    # Pre-populate cache
    cached_payload = {
        "raw_text": '{"summary": "Cached result", "item_count": 3}',
        "parsed": {"summary": "Cached result", "item_count": 3},
        "tokens_in": 50,
        "tokens_out": 20
    }
    write_to_cache(key, cached_payload)

    # Execute call_llm
    result = await call_llm(
        scope=("job", 99),
        phase="research",
        model="test-model",
        messages=messages,
        max_tokens=200,
        schema=SampleOutput,
        db=mock_db
    )

    assert isinstance(result, SampleOutput)
    assert result.summary == "Cached result"
    assert result.item_count == 3

    # Verify ledger was called with cache_hit=True and 0 tokens / 0 USD
    mock_db.add.assert_called_once()
    added_entry = mock_db.add.call_args[0][0]
    assert isinstance(added_entry, LedgerEntry)
    assert added_entry.cache_hit is True
    assert added_entry.tokens_in == 0
    assert added_entry.tokens_out == 0
    assert added_entry.cost_usd_modeled == 0.0
    assert added_entry.latency_ms == 0


# =====================================================================
# 4. Framework Rule: No LangChain Chat Models Allowed
# =====================================================================

def test_framework_rule_no_langchain_model_imports():
    """Unit test asserting no module imports langchain_groq, langchain_openai, etc."""
    import pathlib

    app_dir = pathlib.Path(__file__).resolve().parents[1] / "app"
    forbidden_imports = [
        "langchain_groq",
        "langchain_openai",
        "langchain_anthropic",
        "langchain.chat_models",
    ]

    for py_file in app_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        for forbidden in forbidden_imports:
            assert forbidden not in content, (
                f"Forbidden import '{forbidden}' found in {py_file}. "
                "Framework rule: LangGraph only, no LangChain model wrappers (they bypass call_llm budget)."
            )


@pytest.mark.asyncio
async def test_rate_limiter_single_request_over_limit_does_not_crash():
    """Regression (jobs 15, 17, 18): one estimate > tpm_limit on an empty window raised IndexError on records[0]."""
    limiter = SlidingWindowRateLimiter(tpm_limit=1000, clock_fn=lambda: 1000.0, sleep_fn=AsyncMock())
    await limiter.acquire(estimated_tokens=5000, model="m")
    assert sum(c for _, c in limiter.history["m"]) == 5000  # recorded, so the next call waits for the window
