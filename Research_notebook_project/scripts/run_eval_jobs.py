# scripts/run_eval_jobs.py
"""
Runs the real research jobs used to build the REAL eval dataset and exports everything needed to reproduce
the labeling: claims with snapshot offsets, full source snapshots, shadow verdicts, and provider-measured ledger totals.

Dry run by default (prints the plan). Spends API money only with --yes.
Keys come from .env (GROQ_API_KEY, TAVILY_API_KEY); nothing is logged except ids, counts and hashes.
The LLM disk cache is pointed at a fresh empty directory so no result is a cache replay.
Usage: python scripts/run_eval_jobs.py --split dev [--only q1,q5] [--yes]
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLITS = {"dev": ["q1", "q5", "q6", "q9", "q12", "q13"], "test": ["q2", "q8", "q10", "q14"]}
MAX_TOKENS, MAX_MINUTES, USD_STOP = 120_000, 15, 0.50
OUT = ROOT / "evals" / "real"


def load_env() -> None:
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


async def export_job(qid: str, job_id: int) -> dict:
    from sqlalchemy import select

    from app.db import AsyncSessionLocal
    from app.models import Claim, Job, LedgerEntry, Source, SubQuestion

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        rows = lambda m, col: select(m).where(col == job_id)  # noqa: E731
        sources = (await db.execute(rows(Source, Source.job_id))).scalars().all()
        claims = (await db.execute(rows(Claim, Claim.job_id))).scalars().all()
        subqs = (await db.execute(rows(SubQuestion, SubQuestion.job_id))).scalars().all()
        ledger = (await db.execute(select(LedgerEntry).where(LedgerEntry.scope_id == job_id, LedgerEntry.scope_type == "job"))).scalars().all()
    llm = [x for x in ledger if x.kind == "llm"]
    return {
        "qid": qid, "job_id": job_id, "question": job.question, "status": job.status, "mode": job.mode,
        "subquestions": [{"id": s.id, "text": s.text, "status": s.status} for s in subqs],
        "sources": [{"id": s.id, "url": s.url, "content_hash": s.content_hash, "snapshot": s.snapshot} for s in sources],
        "claims": [{"id": c.id, "subq_id": c.subq_id, "text": c.text, "source_id": c.source_id, "chunk_id": c.chunk_id,
                    "start_char": c.start_char, "end_char": c.end_char, "quote": c.quote, "quote_found": c.quote_found,
                    "entailment": c.entailment, "entailment_reason": c.entailment_reason,
                    "shadow_verdict": c.shadow_verdict, "final_status": c.final_status} for c in claims],
        "ledger": {  # tokens_* are provider-reported only (unknown-usage attempts counted separately); USD is modeled
            "llm_calls": len(llm), "cache_hits": sum(x.cache_hit for x in llm),
            "tokens_in": sum(x.tokens_in for x in llm if x.usage_provenance == "reported"),
            "tokens_out": sum(x.tokens_out for x in llm if x.usage_provenance == "reported"),
            "unknown_usage_attempts": sum(x.usage_provenance == "unknown" and not x.cache_hit for x in llm),
            "failed_attempts": sum(x.status == "failed" for x in llm),
            "usd_modeled": float(sum(x.cost_usd_modeled for x in llm)),
            "latency_ms": [x.latency_ms for x in llm if not x.cache_hit],
        },
    }


async def main(split: str, yes: bool, only: str = "") -> None:
    qs = {q["id"]: q["question"] for q in json.loads((ROOT / "evals" / "benchmark_questions.json").read_text(encoding="utf-8"))}
    ids = [i for i in SPLITS[split] if not only or i in only.split(",")]
    print(f"split={split} jobs={ids} mode=shadow max_tokens={MAX_TOKENS} usd_stop={USD_STOP} (modeled)")
    if not yes:
        # ESTIMATES from the code path (4-8 sub-questions, <=4 claims each), not measurements.
        n = len(ids)
        print(f"ESTIMATE (not measured): ~{35 * n}k-{45 * n}k tokens, ~${0.01 * n:.2f}-${0.02 * n:.2f} modeled, "
              f"~{4 * n}-{8 * n} Tavily searches; hard stop at ${USD_STOP} modeled")
        print("dry run; pass --yes to execute")
        return
    load_env()
    for k in ("GROQ_API_KEY", "TAVILY_API_KEY"):
        if not os.environ.get(k):
            sys.exit(f"{k} missing from .env")
    os.environ["CITEGUARD_CACHE_DIR"] = tempfile.mkdtemp(prefix="citeguard_nocache_")

    from app.db import AsyncSessionLocal, init_db
    from app.models import Job
    from app.pipeline import execute_research_job

    await init_db()
    OUT.mkdir(parents=True, exist_ok=True)
    spent = 0.0
    for qid in ids:
        if (OUT / f"{split}_{qid}.json").exists():
            print(f"{qid}: already exported, skipping")
            continue
        async with AsyncSessionLocal() as db:
            job = Job(question=qs[qid], max_tokens=MAX_TOKENS, max_minutes=MAX_MINUTES, mode="shadow")
            db.add(job)
            await db.commit()
            job_id = job.id
        await execute_research_job(job_id)
        data = await export_job(qid, job_id)
        (OUT / f"{split}_{qid}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
        spent += data["ledger"]["usd_modeled"]
        print(f"{qid}: status={data['status']} claims={len(data['claims'])} sources={len(data['sources'])} "
              f"tokens={data['ledger']['tokens_in'] + data['ledger']['tokens_out']} usd_modeled={data['ledger']['usd_modeled']:.4f}")
        if spent > USD_STOP:
            sys.exit(f"stopping: modeled spend {spent:.2f} > {USD_STOP}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--only", default="", help="comma-separated question ids, e.g. q1,q5")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    asyncio.run(main(a.split, a.yes, a.only))
