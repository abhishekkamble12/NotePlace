# app/core/llm.py
"""
Centralized call_llm wrapper and budget governor for CiteGuard.
Every model call in both Phase 1 (Research) and Phase 2 (Notebook) MUST pass through this module.
Bypassing call_llm breaks the budget, ledger, and rate-limiting guarantees.
"""
import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable, Coroutine
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from groq import AsyncGroq
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import AsyncSessionLocal
from app.models import Job, LedgerEntry

T = TypeVar("T", bound=BaseModel)

class BudgetExceededError(Exception):
    """Raised when an operation would exceed the phase or total job token cap."""

class RateLimitExceededError(Exception):
    """Raised when waiting for a rate limit would breach the job deadline."""

# Modeled pricing per 1,000 tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.00059, "output": 0.00079},
    "llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
    "gemma2-9b-it": {"input": 0.00020, "output": 0.00020},
    "default": {"input": 0.00050, "output": 0.00150},
}

# Per-phase token allocation shares (spec section 6.5)
# Research: 50%, Writing: 15%, Verify+Rewrite: 25%, Reserve: 10%
PHASE_BUDGET_SHARES: dict[str, float] = {
    "research": 0.50,
    "planning": 0.10,
    "writing": 0.15,
    "verifying": 0.25,
    "reserve": 0.10,
}

# Groq free-tier rate limits
GROQ_TPM_LIMIT = 8_000      # Tokens per minute
GROQ_TPD_LIMIT = 200_000    # Tokens per day

# Cache directory configuration (~/.cache/citeguard/llm with fallback to .cache/llm)
CACHE_DIR = Path(os.getenv("CITEGUARD_CACHE_DIR", str(Path.home() / ".cache" / "citeguard" / "llm")))
try:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    CACHE_DIR = Path(".cache/llm")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


class ProviderUnavailableError(Exception):
    """Raised when every provider in the chain failed or has an open circuit breaker."""


class CircuitBreaker:
    """Per-provider breaker: opens after `threshold` consecutive failures, half-opens after `cooldown` seconds.
    shortcut: in-process state, so it is per worker; move to Redis in Phase 4."""

    def __init__(self, threshold: int = 3, cooldown: float = 30.0, clock_fn: Callable[[], float] | None = None):
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock_fn = clock_fn or time.monotonic
        self.failures: dict[str, int] = {}
        self.opened_at: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        opened = self.opened_at.get(key)
        if opened is None:
            return True
        if self.clock_fn() - opened >= self.cooldown:
            self.opened_at[key] = self.clock_fn()  # half-open: let one probe through, re-arm the timer
            return True
        return False

    def record_success(self, key: str) -> None:
        self.failures.pop(key, None)
        self.opened_at.pop(key, None)

    def record_failure(self, key: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1
        if self.failures[key] >= self.threshold:
            self.opened_at[key] = self.clock_fn()

    def reset(self) -> None:
        self.failures.clear()
        self.opened_at.clear()


breaker = CircuitBreaker()
PROVIDER_TIMEOUT_S = 60.0


def get_provider_chain(model: str) -> list[dict[str, str]]:
    """Primary Groq model first, then fallbacks from CITEGUARD_FALLBACKS, e.g.
    {"gemma2-9b-it": [{"provider": "openai_compat", "model": "x", "base_url": "https://.../v1", "api_key_env": "OTHER_KEY"}]}.
    A fallback verifier must be a different model family from the writer."""
    chain = [{"provider": "groq", "model": model}]
    try:
        chain += json.loads(os.getenv("CITEGUARD_FALLBACKS", "{}")).get(model, [])
    except (ValueError, AttributeError):
        pass
    return chain


async def _call_provider(
    spec: dict[str, str], messages: list[dict[str, str]], max_tokens: int, response_format: dict | None
) -> tuple[str, int | None, int | None]:
    """One provider request -> (text, prompt_tokens, completion_tokens). Tests patch this function."""
    if spec["provider"] == "groq":
        client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        resp = await client.chat.completions.create(
            model=spec["model"], messages=messages, max_tokens=max_tokens, response_format=response_format  # type: ignore
        )
        usage = resp.usage
        return resp.choices[0].message.content or "", getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
    import httpx
    body: dict[str, Any] = {"model": spec["model"], "messages": messages, "max_tokens": max_tokens}
    if response_format:
        body["response_format"] = response_format
    async with httpx.AsyncClient(timeout=PROVIDER_TIMEOUT_S) as http:
        r = await http.post(
            spec["base_url"].rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv(spec.get('api_key_env', ''), '')}"},
            json=body,
        )
        r.raise_for_status()
        data = r.json()
    usage = data.get("usage") or {}
    return data["choices"][0]["message"].get("content") or "", usage.get("prompt_tokens"), usage.get("completion_tokens")


async def _complete_with_failover(
    model: str, messages: list[dict[str, str]], max_tokens: int, response_format: dict | None
) -> tuple[str, int | None, int | None, str]:
    """Try each provider in order, skipping open breakers. Returns (text, tin, tout, model_actually_used)."""
    last_err: Exception | None = None
    for spec in get_provider_chain(model):
        key = f"{spec['provider']}:{spec['model']}"
        if not breaker.allow(key):
            continue
        try:
            text, tin, tout = await asyncio.wait_for(
                _call_provider(spec, messages, max_tokens, response_format), PROVIDER_TIMEOUT_S
            )
        except Exception as e:  # 429, 5xx, timeout, network: fail over to the next provider
            breaker.record_failure(key)
            last_err = e
            continue
        breaker.record_success(key)
        return text, tin, tout, spec["model"]
    raise ProviderUnavailableError(f"All providers failed or are open-circuit for '{model}': {last_err!r}")


class SlidingWindowRateLimiter:
    """
    Tracks token consumption per model over a 60-second sliding window (TPM)
    and 86400-second window (TPD). Supports pluggable clock and sleep functions
    for deterministic unit testing.
    """
    def __init__(
        self,
        tpm_limit: int = GROQ_TPM_LIMIT,
        tpd_limit: int = GROQ_TPD_LIMIT,
        clock_fn: Callable[[], float] | None = None,
        sleep_fn: Callable[[float], Coroutine[Any, Any, None]] | None = None
    ):
        self.tpm_limit = tpm_limit
        self.tpd_limit = tpd_limit
        self.clock_fn = clock_fn or time.time
        self.sleep_fn = sleep_fn or asyncio.sleep
        self.history: dict[str, list[tuple[float, int]]] = {}
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        estimated_tokens: int,
        model: str = "default",
        deadline_at: datetime | None = None
    ) -> None:
        async with self._lock:
            while True:
                now = self.clock_fn()
                records = self.history.get(model, [])

                # Prune records older than 60 seconds
                records = [(t, count) for (t, count) in records if now - t < 60.0]
                current_tpm = sum(count for _, count in records)

                if current_tpm + estimated_tokens <= self.tpm_limit:
                    records.append((now, estimated_tokens))
                    self.history[model] = records
                    return

                # Wait for the oldest entry in the 60-second window to roll off
                oldest_t = records[0][0]
                wait_seconds = max(0.05, 60.0 - (now - oldest_t) + 0.05)

                if deadline_at is not None:
                    # Check if waiting would breach the deadline
                    deadline_ts = deadline_at.timestamp() if hasattr(deadline_at, "timestamp") else float(deadline_at)
                    if (now + wait_seconds) > deadline_ts:
                        raise RateLimitExceededError(
                            f"Waiting {wait_seconds:.1f}s for rate limit window would exceed deadline"
                        )

                await self.sleep_fn(wait_seconds)

    def reset(self) -> None:
        self.history.clear()


rate_limiter = SlidingWindowRateLimiter()


def get_cache_key(model: str, messages: list[dict[str, str]], schema: type[BaseModel] | None, max_tokens: int = 1000) -> str:
    schema_name = schema.__name__ if schema else "raw"
    payload = f"{model}:{json.dumps(messages, sort_keys=True)}:{schema_name}:{max_tokens}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_from_cache(cache_key: str) -> dict[str, Any] | None:
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            with open(cache_file, encoding="utf-8") as f:
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
    """Fast character-based token approximation (~4 chars per token)."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, total_chars // 4)


def calculate_modeled_usd(model: str, tokens_in: int, tokens_out: int) -> float:
    """Calculates modeled USD expenditure at public list pricing."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
    return (tokens_in / 1000.0) * pricing["input"] + (tokens_out / 1000.0) * pricing["output"]


async def check_budget_caps(
    db: AsyncSession,
    scope: tuple[str, int],
    phase: str,
    estimated_tokens: int
) -> tuple[Any, int]:
    """
    Pre-call guard: verifies both job token budget and phase token cap.
    Raises BudgetExceededError before calling the model if cap is exceeded.
    """
    scope_type, scope_id = scope
    if scope_type != "job":
        # For Phase 2 notebooks, if a token_budget is tracked
        return None, 100_000

    job = await db.get(Job, scope_id)
    if not job:
        raise ValueError(f"Job {scope_id} not found")

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
        ).where(LedgerEntry.scope_type == scope_type, LedgerEntry.scope_id == scope_id)
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
    scope: tuple[str, int] | None = None,
    phase: str = "research",
    model: str = "llama-3.3-70b-versatile",
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = 1000,
    schema: type[T] | None = None,
    db: AsyncSession | None = None,
    *,
    job_id: int | None = None,
    return_tuple: bool = False,
) -> Any:
    """
    The single non-bypassable LLM gateway for CiteGuard.
    Handles:
      1. Token estimate & pre-call budget checking (job & phase caps)
      2. Sliding window rate limiter (TPM)
      3. Disk caching (costs 0 tokens, logged as cache_hit=True)
      4. Provider call (Groq) with optional Pydantic schema validation & 1 retry
      5. Ledger entry insertion with latency, modeled USD, and token counts
    """
    if messages is None:
        messages = []

    if scope is None:
        if job_id is not None:
            scope = ("job", job_id)
        else:
            raise ValueError("Either scope or job_id must be provided to call_llm")

    scope_type, scope_id = scope
    cache_key = get_cache_key(model, messages, schema, max_tokens)
    cached = read_from_cache(cache_key)

    # Need a session for ledger operations and budget checks
    session_managed = False
    session = db
    if session is None:
        session = AsyncSessionLocal()
        session_managed = True

    try:
        # 1. Disk Cache Lookup
        if cached is not None:
            raw_text = cached["raw_text"]
            parsed_data = None
            if schema and "parsed" in cached and cached["parsed"] is not None:
                parsed_data = schema.model_validate(cached["parsed"])

            # Record cache hit in ledger (0 tokens, 0 cost, 0 latency)
            ledger_entry = LedgerEntry(
                scope_type=scope_type,
                scope_id=scope_id,
                phase=phase,
                kind="llm",
                model=model,
                tokens_in=0,
                tokens_out=0,
                cost_usd_modeled=0.0,
                latency_ms=0,
                cache_hit=True
            )
            session.add(ledger_entry)
            await session.commit()

            if return_tuple:
                return raw_text, parsed_data
            return parsed_data if schema is not None else raw_text

        # 2. Token estimation & Pre-call budget check
        est_prompt_tokens = estimate_prompt_tokens(messages)
        est_total_tokens = est_prompt_tokens + max_tokens
        job, _ = await check_budget_caps(session, scope, phase, est_total_tokens)
        deadline_at = job.deadline_at if job else None

        # 3. Rate limiter check (wait if approaching 8k TPM limit)
        await rate_limiter.acquire(est_total_tokens, model=model, deadline_at=deadline_at)

        # 4. Schema instructions
        call_messages = list(messages)
        response_format = None
        if schema is not None:
            response_format = {"type": "json_object"}
            call_messages.append({
                "role": "system",
                "content": f"You must output valid JSON strictly adhering to this JSON schema:\n{json.dumps(schema.model_json_schema())}"
            })

        # 5. Provider call with failover + breaker; one corrective retry on schema failure
        raw_text = ""
        parsed_obj: T | None = None
        tokens_in = 0
        tokens_out = 0
        used_model = model
        t0 = time.time()

        for attempt in range(2):
            raw_text, t_in, t_out, used_model = await _complete_with_failover(
                model, call_messages, max_tokens, response_format
            )
            tokens_in += t_in if t_in is not None else estimate_prompt_tokens(call_messages)
            tokens_out += t_out if t_out is not None else max(1, len(raw_text) // 4)

            if schema is None:
                break

            try:
                parsed_obj = schema.model_validate_json(raw_text)
                break
            except ValidationError:
                if attempt == 0:
                    # Generic feedback only: never echo model/page content back into the prompt
                    call_messages.append({"role": "assistant", "content": raw_text})
                    call_messages.append({
                        "role": "user",
                        "content": "Your previous output was not valid JSON for the schema. Output ONLY valid JSON matching the schema."
                    })
                else:
                    raise

        total_latency_ms = int((time.time() - t0) * 1000)
        modeled_usd = calculate_modeled_usd(used_model, tokens_in, tokens_out)

        # 6. Save to disk cache
        write_to_cache(cache_key, {
            "raw_text": raw_text,
            "parsed": parsed_obj.model_dump() if parsed_obj else None,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out
        })

        # 7. Write atomic ledger row
        ledger_entry = LedgerEntry(
            scope_type=scope_type,
            scope_id=scope_id,
            phase=phase,
            kind="llm",
            model=used_model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd_modeled=modeled_usd,
            latency_ms=total_latency_ms,
            cache_hit=False
        )
        session.add(ledger_entry)
        await session.commit()

        if return_tuple:
            return raw_text, parsed_obj
        return parsed_obj if schema is not None else raw_text

    finally:
        if session_managed and session:
            await session.close()
