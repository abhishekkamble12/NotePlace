"""Chaos tests: the REAL call_llm, the REAL Groq SDK and a REAL Postgres ledger talk over HTTP to a scriptable fake provider.
Nothing in app/ is patched. Config only: GROQ_BASE_URL, cache dir, limiter size, provider timeout.
Fake-provider responses are SYNTHETIC; these prove failure handling and accounting, not model quality."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select

from app.core import llm
from app.core.accounting import ledger_summary
from app.core.llm import BudgetExceededError, ProviderUnavailableError, call_llm, spend_expr
from app.models import Job, LedgerEntry

pytestmark = [pytest.mark.chaos, pytest.mark.int]


class Verdict(BaseModel):
    verdict: str


def completion(content, pt=11, ct=3):
    return json.dumps({"id": "x", "object": "chat.completion", "created": 1, "model": "m", "choices": [
        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}).encode()


class Fake:
    """script: list of behaviours consumed one per request on the groq path; the last one repeats. fallback path always succeeds."""

    def __init__(self, script):
        self.script, self.hits, self.fallback_hits = list(script), 0, 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.startswith("/v1/"):
                    outer.fallback_hits += 1
                    return self._send(200, completion('{"verdict": "supports"}', 7, 2))
                outer.hits += 1
                b = outer.script[min(outer.hits - 1, len(outer.script) - 1)]
                kind = b[0]
                if kind == "ok":
                    self._send(200, completion(b[1], *b[2:]))
                elif kind == "status":
                    self._send(b[1], b'{"error": {"message": "boom"}}', {"retry-after-ms": "1"})
                elif kind == "hang":
                    threading.Event().wait(b[1])
                elif kind == "slowok":
                    threading.Event().wait(b[1])
                    self._send(200, completion(b[2], *b[3:]))

            def _send(self, code, body, extra=None):
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    for k, v in (extra or {}).items():
                        self.send_header(k, v)
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def env(sm_factory, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm.rate_limiter, "tpm_limit", 10**9)  # config, not a patch: TPM waiting is covered by unit tests
    monkeypatch.setenv("GROQ_API_KEY", "k")
    monkeypatch.delenv("CITEGUARD_FALLBACKS", raising=False)
    llm.breaker.reset()
    llm.rate_limiter.reset()
    made = []

    def start(script):
        f = Fake(script)
        made.append(f)
        monkeypatch.setenv("GROQ_BASE_URL", f.url)
        return f

    yield sm_factory(), start, monkeypatch
    for f in made:
        f.close()


async def mkjob(sm, max_tokens=100_000):
    async with sm() as s:
        j = Job(question="q", max_tokens=max_tokens)
        s.add(j)
        await s.commit()
        return j.id


async def go(sm, jid, tag, schema=None, phase="research", max_tokens=200):
    async with sm() as s:
        return await call_llm(scope=("job", jid), phase=phase, model="m", messages=[{"role": "user", "content": tag}],
                              max_tokens=max_tokens, schema=schema, db=s)


async def rows(sm, jid):
    async with sm() as s:
        return (await s.execute(select(LedgerEntry).where(LedgerEntry.scope_id == jid).order_by(LedgerEntry.id))).scalars().all()


@pytest.mark.asyncio
async def test_happy_path_ledger_matches_provider_usage(env):
    sm, start, _ = env
    start([("ok", "hello", 123, 45)])
    jid = await mkjob(sm)
    assert await go(sm, jid, "a") == "hello"
    (r,) = await rows(sm, jid)
    assert (r.status, r.tokens_in, r.tokens_out, r.usage_provenance, r.cache_hit) == ("ok", 123, 45, "reported", False)


@pytest.mark.asyncio
async def test_provider_429_forever_raises_typed_error_and_leaves_no_phantom_billing(env):
    sm, start, _ = env
    f = start([("status", 429)])
    jid = await mkjob(sm)
    with pytest.raises(ProviderUnavailableError):
        await go(sm, jid, "a")
    assert f.hits >= 1
    rs = await rows(sm, jid)
    assert rs and all(r.status == "failed" and not r.cache_hit for r in rs)
    async with sm() as s:
        u = await ledger_summary(s, "job", jid)
    assert u["provider_reported_tokens"] == 0  # nothing was reported by a 429


@pytest.mark.asyncio
async def test_transient_500_then_success_returns_the_answer(env):
    sm, start, _ = env
    start([("status", 500), ("ok", "recovered", 9, 1)])
    jid = await mkjob(sm)
    assert await go(sm, jid, "a") == "recovered"
    assert [r.status for r in await rows(sm, jid)][-1] == "ok"


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["not json at all", "", '{"verdict": 5, "extra": 1}', "```json\n{}\n```"])
async def test_malformed_or_empty_output_fails_validation_after_exactly_two_billed_attempts(env, content):
    sm, start, _ = env
    f = start([("ok", content, 30, 10)])
    jid = await mkjob(sm)
    with pytest.raises(ValidationError):
        await go(sm, jid, "a", schema=Verdict)
    assert f.hits == 2
    rs = await rows(sm, jid)
    assert len(rs) == 2 and all(r.status == "ok" and r.tokens_in == 30 for r in rs)  # both attempts billed, not hidden


@pytest.mark.asyncio
async def test_malformed_first_then_valid_second_succeeds(env):
    sm, start, _ = env
    start([("ok", "garbage", 5, 5), ("ok", '{"verdict": "supports"}', 6, 6)])
    jid = await mkjob(sm)
    assert (await go(sm, jid, "a", schema=Verdict)).verdict == "supports"
    assert len(await rows(sm, jid)) == 2


@pytest.mark.asyncio
async def test_hung_provider_times_out_keeps_the_reservation_and_opens_no_phantom_success(env):
    sm, start, mp = env
    start([("hang", 5)])
    mp.setattr(llm, "PROVIDER_TIMEOUT_S", 0.3)
    jid = await mkjob(sm)
    with pytest.raises(ProviderUnavailableError):
        await go(sm, jid, "a")
    (r,) = await rows(sm, jid)
    assert (r.status, r.usage_known, r.usage_provenance) == ("failed", False, "unknown") and r.reserved_tokens > 0


@pytest.mark.asyncio
async def test_primary_down_falls_over_to_second_provider_and_both_attempts_are_in_the_ledger(env):
    sm, start, mp = env
    f = start([("status", 503)])
    mp.setenv("CITEGUARD_FALLBACKS", json.dumps({"m": [{"provider": "openai_compat", "model": "m2", "base_url": f.url + "/v1", "api_key_env": "GROQ_API_KEY"}]}))
    jid = await mkjob(sm)
    assert (await go(sm, jid, "a", schema=Verdict)).verdict == "supports"
    rs = await rows(sm, jid)
    assert [(r.provider, r.status) for r in rs] == [("groq", "failed"), ("openai_compat", "ok")]
    assert len({r.call_id for r in rs}) == 1 and f.fallback_hits == 1


@pytest.mark.asyncio
async def test_circuit_breaker_stops_hammering_a_dead_provider(env):
    sm, start, _ = env
    f = start([("status", 503)])
    jid = await mkjob(sm)
    for i in range(3):
        with pytest.raises(ProviderUnavailableError):
            await go(sm, jid, f"c{i}")
    hits_when_open = f.hits
    with pytest.raises(ProviderUnavailableError):
        await go(sm, jid, "after-open")
    assert f.hits == hits_when_open, "breaker is open: no further request may reach the dead provider"


@pytest.mark.asyncio
async def test_concurrent_calls_over_real_http_never_exceed_the_budget(env):
    sm, start, _ = env
    start([("slowok", 2.0, "x", 1, 100)])  # usage within the reservation (prompt est ~1 + max_tokens 200)
    jid = await mkjob(sm, max_tokens=4_000)  # research cap 2000; each call reserves ~201, so at most 9 may be in flight
    res = await asyncio.gather(*[go(sm, jid, f"p{i}") for i in range(30)], return_exceptions=True)
    ok = [r for r in res if r == "x"]
    refused = [r for r in res if isinstance(r, BudgetExceededError)]
    assert len(ok) + len(refused) == 30 and refused, "the cap must have bitten"
    async with sm() as s:
        spent = (await s.execute(select(func.coalesce(func.sum(spend_expr()), 0)).where(LedgerEntry.scope_id == jid))).scalar()
    assert spent <= 2000
    assert spent == len(ok) * 101  # settled spend equals exactly the provider-reported tokens of the calls that ran


@pytest.mark.asyncio
async def test_job_row_lock_is_not_left_held_after_provider_failure(env):
    sm, start, _ = env
    start([("status", 500)])
    jid = await mkjob(sm)
    with pytest.raises(ProviderUnavailableError):
        await go(sm, jid, "a")
    async with sm() as s:
        await asyncio.wait_for(s.get(Job, jid, with_for_update=True), 5)


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, reason="DESIGN: the cap is enforced on a chars/4 estimate; provider-reported usage above the reservation is accepted after the fact, so settled spend can pass the cap")
async def test_settled_spend_never_exceeds_cap_even_when_provider_reports_more_than_estimated(env):
    sm, start, _ = env
    start([("slowok", 2.0, "x", 400, 100)])  # provider says 500 tokens for a tiny prompt the estimator priced at ~201
    jid = await mkjob(sm, max_tokens=4_000)
    await asyncio.gather(*[go(sm, jid, f"p{i}") for i in range(30)], return_exceptions=True)
    async with sm() as s:
        spent = (await s.execute(select(func.coalesce(func.sum(spend_expr()), 0)).where(LedgerEntry.scope_id == jid))).scalar()
    assert spent <= 2000
