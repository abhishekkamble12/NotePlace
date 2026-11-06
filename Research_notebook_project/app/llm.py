# app/llm.py
"""
Backwards-compatibility re-export module.
All core LLM functionality lives in app.core.llm.
"""
from app.core.llm import (
    CACHE_DIR,
    GROQ_TPD_LIMIT,
    GROQ_TPM_LIMIT,
    MODEL_PRICING,
    PHASE_BUDGET_SHARES,
    BudgetExceededError,
    RateLimitExceededError,
    SlidingWindowRateLimiter,
    calculate_modeled_usd,
    call_llm,
    check_budget_caps,
    estimate_prompt_tokens,
    get_cache_key,
    rate_limiter,
    read_from_cache,
    write_to_cache,
)

__all__ = [
    "CACHE_DIR",
    "GROQ_TPD_LIMIT",
    "GROQ_TPM_LIMIT",
    "MODEL_PRICING",
    "PHASE_BUDGET_SHARES",
    "BudgetExceededError",
    "RateLimitExceededError",
    "SlidingWindowRateLimiter",
    "calculate_modeled_usd",
    "call_llm",
    "check_budget_caps",
    "estimate_prompt_tokens",
    "get_cache_key",
    "rate_limiter",
    "read_from_cache",
    "write_to_cache",
]
