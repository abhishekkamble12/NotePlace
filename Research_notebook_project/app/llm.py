# app/llm.py
import os
import json
import time
import hashlib
import asyncio
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, TypeVar
from pydantic import BaseModel, ValidationError
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from groq import AsyncGroq
from app.models import Job, LedgerEntry

T = TypeVar("T", bound=BaseModel)

class BudgetExceededError(Exception):
    """Raised when an operation would exceed the phase or total job token/USD cap."""
    pass

class RateLimitExceededError(Exception):
    """Raised when waiting for a rate limit would breach the job's deadline."""
    pass

# Modeled pricing per 1,000 tokens (USD)
MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "gemma2-9b-it": {"input": 0.00020, "output": 0.00020},
    "default": {"input": 0.00050, "output": 0.00150},
}

# Per-phase token allocation
PHASE_BUDGET_SHARES = {
    "planning": 0.10,
    "research": 0.40,
    "writing": 0.15,
    "verifying": 0.25,
    "reserve": 0.10,
}

# Groq free-tier rate limits
GROQ_TPM_LIMIT = 8_000      # Tokens per minute
GROQ_TPD_LIMIT = 200_000    # Tokens per day

# Disk cache directory
CACHE_DIR = Path(".cache/llm")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class SlidingWindowRateLimiter:
    """Tracks token consumption over a 60-second sliding window."""
    def __init__(self, tpm_limit: int = GROQ_TPM_LIMIT):
        self.tpm_limit = tpm_limit
        self.history: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int, deadline_at: datetime | None = None) -> None:
        async with self._lock:
            while True:
                now = time.time()
                # Prune records older than 60s
                self.history = [(t, count) for (t, count) in self.history if now - t < 60.0]
                current_tpm = sum(count for _, count in self.history)

                if current_tpm + estimated_tokens <= self.tpm_limit:
                    self.history.append((now, estimated_tokens))
                    return

                # Need to wait for oldest entry to age out
                oldest_t = self.history[0][0]
                wait_seconds = max(0.1, 60.0 - (now - oldest_t) + 0.1)

                if deadline_at is not None:
                    # Check if waiting breaches deadline
                    now_utc = datetime.now(timezone.utc)
                    if (now_utc.timestamp() + wait_seconds) > deadline_at.timestamp():
                        raise RateLimitExceededError(
                            f"Waiting {wait_seconds:.1f}s for rate limit window would exceed deadline"
                        )

                await asyncio.sleep(wait_seconds)

rate_limiter = SlidingWindowRateLimiter()


def get_cache_key(model: str, messages: list[dict[str, str]], schema: type[BaseModel] | None) -> str:
    schema_name = schema.__name__ if schema else "raw"
    payload = f"{model}:{json.dumps(messages, sort_keys=True)}:{schema_name}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_from_cache(cache_key: str) -> dict[str, Any] | None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def write_to_cache(cache_key: str, data: dict[str, Any]) -> None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    try:
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    """Fast token approximation (~4 chars per token)."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return max(1, total_chars // 4)


def calculate_modeled_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    return (tokens_in / 1000.0) * pricing["input"] + (tokens_out / 1000.0) * pricing["output"]


async def check_budget_caps(
    db: AsyncSession,
    job_id: int,
    phase: str,
    estimated_tokens: int
) -> tuple[Job, float]:
    """Pre-call guard: verifies both job token budget and phase token cap."""
    job = await db.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    max_tokens = job.max_tokens
    phase_share = PHASE_BUDGET_SHARES.get(phase, 0.25)
    phase_cap = int(max_tokens * phase_share)

    # Sum non-cached tokens spent on this job
    result = await db.execute(
        select(
            func.coalesce(
                func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out).filter(LedgerEntry.cache_hit.is_(False)),
                0
            ).label("total_tokens"),
            func.coalesce(
                func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out).filter(
                    LedgerEntry.phase == phase,
                    LedgerEntry.cache_hit.is_(False)
                ),
                0
            ).label("phase_tokens"),
        ).where(LedgerEntry.job_id == job_id)
    )
    row = result.one()
    total_tokens = int(row.total_tokens)
    phase_tokens = int(row.phase_tokens)

    if phase_tokens + estimated_tokens > phase_cap:
        raise BudgetExceededError(
            f"Phase '{phase}' token cap reached: used {phase_tokens} + est {estimated_tokens} > cap {phase_cap}"
        )

    if total_tokens + estimated_tokens > max_tokens:
        raise BudgetExceededError(
            f"Job token budget exceeded: used {total_tokens} + est {estimated_tokens} > max {max_tokens}"
        )

    return job, phase_cap


async def call_llm(
    db: AsyncSession,
    job_id: int,
    phase: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 1000,
    schema: type[T] | None = None
) -> tuple[str, T | None]:
    """
    The single, non-bypassable LLM gateway for CiteGuard.
    Handles:
      1. Token estimate & pre-call budget checking
      2. Sliding window rate limiter (TPM)
      3. Disk caching (costs 0 tokens on hit)
      4. Provider call (Groq) with optional Pydantic schema validation & 1 retry
      5. Ledger entry insertion with latency, modeled USD, and token counts
    """
    cache_key = get_cache_key(model, messages, schema)
    cached = read_from_cache(cache_key)

    # 1. Check Disk Cache first
    if cached is not None:
        raw_text = cached["raw_text"]
        parsed_data = None
        if schema and "parsed" in cached and cached["parsed"] is not None:
            parsed_data = schema.model_validate(cached["parsed"])

        # Record cache hit in ledger (0 tokens, 0 cost, 0 latency)
        ledger_entry = LedgerEntry(
            job_id=job_id,
            phase=phase,
            kind="llm",
            model=model,
            tokens_in=0,
            tokens_out=0,
            cost_usd_modeled=0.0,
            latency_ms=0,
            cache_hit=True
        )
        db.add(ledger_entry)
        await db.commit()
        return raw_text, parsed_data

    # 2. Estimate tokens and check budget caps
    est_prompt_tokens = estimate_prompt_tokens(messages)
    est_total_tokens = est_prompt_tokens + max_tokens
    job, _ = await check_budget_caps(db, job_id, phase, est_total_tokens)

    # 3. Rate limiter check (wait if approaching 8k TPM limit)
    await rate_limiter.acquire(est_total_tokens, deadline_at=job.deadline_at)

    # 4. Prepare Client and Schema instructions
    api_key = os.getenv("GROQ_API_KEY")
    client = AsyncGroq(api_key=api_key)

    call_messages = list(messages)
    response_format = None
    if schema is not None:
        response_format = {"type": "json_object"}
        call_messages.append({
            "role": "system",
            "content": f"You must output valid JSON strictly adhering to this JSON schema:\n{json.dumps(schema.model_json_schema())}"
        })

    # 5. Call API with 1 retry on Pydantic validation failure
    raw_text = ""
    parsed_obj: T | None = None
    tokens_in = est_prompt_tokens
    tokens_out = 0
    t0 = time.time()

    for attempt in range(2):
        t_call_start = time.time()
        response = await client.chat.completions.create(
            model=model,
            messages=call_messages, # type: ignore
            max_tokens=max_tokens,
            response_format=response_format # type: ignore
        )
        latency_ms = int((time.time() - t_call_start) * 1000)

        choice = response.choices[0]
        raw_text = choice.message.content or ""
        if response.usage:
            tokens_in = response.usage.prompt_tokens
            tokens_out = response.usage.completion_tokens

        if schema is None:
            break

        try:
            parsed_obj = schema.model_validate_json(raw_text)
            break
        except ValidationError as ve:
            if attempt == 0:
                # Add retry feedback message to correct invalid JSON
                call_messages.append({"role": "assistant", "content": raw_text})
                call_messages.append({
                    "role": "user",
                    "content": f"JSON validation failed with error: {ve}. Please correct the output strictly following the schema."
                })
            else:
                raise ve

    total_latency_ms = int((time.time() - t0) * 1000)
    modeled_usd = calculate_modeled_usd(model, tokens_in, tokens_out)

    # 6. Save to disk cache
    write_to_cache(cache_key, {
        "raw_text": raw_text,
        "parsed": parsed_obj.model_dump() if parsed_obj else None,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out
    })

    # 7. Write atomic ledger row
    ledger_entry = LedgerEntry(
        job_id=job_id,
        phase=phase,
        kind="llm",
        model=model,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd_modeled=modeled_usd,
        latency_ms=total_latency_ms,
        cache_hit=False
    )
    db.add(ledger_entry)
    await db.commit()

    return raw_text, parsed_obj

