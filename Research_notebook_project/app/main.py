# app/main.py
import csv
import io
import asyncio
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db, init_db
from app.models import Job, Claim, LedgerEntry, Source
from app.pipeline import execute_research_job

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when server starts: ensures pgvector & all tables exist
    await init_db()
    yield

app = FastAPI(
    title="CiteGuard API",
    description="Budget-Bounded Research Service with Verified Citations",
    version="0.2.0",
    lifespan=lifespan
)

class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, description="The research question to investigate")
    max_tokens: int = Field(default=50_000, description="Hard token budget cap")
    max_usd: float = Field(default=0.50, description="Modeled USD budget ceiling")
    max_minutes: int = Field(default=8, description="Time deadline in minutes")
    mode: Literal["single", "multi", "shadow"] = Field(
        default="multi",
        description="Execution mode: multi (parallel), single (sequential), or shadow (records verifier verdicts without dropping)"
    )

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(1))
    return {"status": "ok", "db": result.scalar() == 1}

@app.post("/research", status_code=202)
async def create_research_job(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Submits a new research job. Spawns an async research pipeline.
    """
    job = Job(
        question=request.question,
        max_tokens=request.max_tokens,
        max_usd=request.max_usd,
        max_minutes=request.max_minutes,
        mode=request.mode,
        status="queued",
        stage="planning"
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Launch pipeline in background task
    background_tasks.add_task(execute_research_job, job.id)

    return {"job_id": job.id, "status": "queued"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Aggregate spend from ledger
    ledger_res = await db.execute(
        select(
            func.coalesce(func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out), 0).label("tokens_used"),
            func.coalesce(func.sum(LedgerEntry.cost_usd_modeled), 0.0).label("usd_spent")
        ).where(LedgerEntry.job_id == job_id)
    )
    tokens_used, usd_spent = ledger_res.one()

    # Count claims progress
    claims_res = await db.execute(
        select(
            func.count(Claim.id).label("total_claims"),
            func.count(Claim.id).filter(Claim.final_status == "kept").label("kept_claims")
        ).where(Claim.job_id == job_id)
    )
    total_claims, kept_claims = claims_res.one()

    return {
        "job_id": job.id,
        "question": job.question,
        "status": job.status,
        "stage": job.stage,
        "mode": job.mode,
        "tokens_spent": int(tokens_used),
        "tokens_budget": job.max_tokens,
        "usd_spent": float(usd_spent),
        "total_claims": total_claims,
        "verified_claims": kept_claims,
        "created_at": job.created_at
    }

@app.get("/jobs/{job_id}/report")
async def get_job_report(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.report_md:
        return {"job_id": job.id, "status": job.status, "stage": job.stage, "report": "Report is still being generated."}

    return Response(content=job.report_md, media_type="text/markdown")

@app.get("/jobs/{job_id}/claims")
async def get_job_claims(job_id: int, db: AsyncSession = Depends(get_db)):
    claims_res = await db.execute(
        select(Claim).where(Claim.job_id == job_id)
    )
    claims = claims_res.scalars().all()
    return [
        {
            "id": c.id,
            "claim": c.text,
            "quote": c.quote,
            "quote_found": c.quote_found,
            "entailment": c.entailment,
            "entailment_reason": c.entailment_reason,
            "final_status": c.final_status,
            "confidence": c.confidence,
            "shadow_verdict": c.shadow_verdict,
            "rewrite_count": c.rewrite_count
        }
        for c in claims
    ]

@app.get("/jobs/{job_id}/ledger")
async def get_job_ledger(job_id: int, db: AsyncSession = Depends(get_db)):
    ledger_res = await db.execute(
        select(LedgerEntry).where(LedgerEntry.job_id == job_id).order_by(LedgerEntry.id.asc())
    )
    entries = ledger_res.scalars().all()
    return [
        {
            "id": e.id,
            "phase": e.phase,
            "model": e.model,
            "tokens_in": e.tokens_in,
            "tokens_out": e.tokens_out,
            "cost_usd_modeled": float(e.cost_usd_modeled),
            "latency_ms": e.latency_ms,
            "cache_hit": e.cache_hit,
            "created_at": e.created_at
        }
        for e in entries
    ]

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.status = "cancelled"
    await db.commit()
    return {"job_id": job.id, "status": "cancelled"}

@app.get("/export/claims.csv")
async def export_claims_csv(db: AsyncSession = Depends(get_db)):
    """Exports all claims to CSV for human labeling and evaluation."""
    claims_res = await db.execute(select(Claim, Job.question).join(Job, Claim.job_id == Job.id))
    rows = claims_res.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "claim_id", "job_id", "question", "claim_text", "exact_quote",
        "quote_found", "entailment_verdict", "final_status", "human_label"
    ])

    for claim, question in rows:
        writer.writerow([
            claim.id, claim.job_id, question, claim.text, claim.quote,
            claim.quote_found, claim.entailment, claim.final_status, claim.human_label
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=claims_evaluation.csv"}
    )


@app.post("/research", status_code=202)
async def create_research_job(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """
    Submits a new research job. Spawns an async research pipeline.
    """
    job = Job(
        question=request.question,
        max_tokens=request.max_tokens,
        max_usd=request.max_usd,
        max_minutes=request.max_minutes,
        mode=request.mode,
        status="queued",
        stage="planning"
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Launch pipeline in background task
    background_tasks.add_task(execute_research_job, job.id)

    return {"job_id": job.id, "status": "queued"}

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Aggregate spend from ledger
    ledger_res = await db.execute(
        select(
            func.coalesce(func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out), 0).label("tokens_used"),
            func.coalesce(func.sum(LedgerEntry.cost_usd_modeled), 0.0).label("usd_spent")
        ).where(LedgerEntry.job_id == job_id)
    )
    tokens_used, usd_spent = ledger_res.one()

    # Count claims progress
    claims_res = await db.execute(
        select(
            func.count(Claim.id).label("total_claims"),
            func.count(Claim.id).filter(Claim.final_status == "kept").label("kept_claims")
        ).where(Claim.job_id == job_id)
    )
    total_claims, kept_claims = claims_res.one()

    return {
        "job_id": job.id,
        "question": job.question,
        "status": job.status,
        "stage": job.stage,
        "mode": job.mode,
        "tokens_spent": int(tokens_used),
        "tokens_budget": job.max_tokens,
        "usd_spent": float(usd_spent),
        "total_claims": total_claims,
        "verified_claims": kept_claims,
        "created_at": job.created_at
    }

@app.get("/jobs/{job_id}/report")
async def get_job_report(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if not job.report_md:
        return {"job_id": job.id, "status": job.status, "stage": job.stage, "report": "Report is still being generated."}

    return Response(content=job.report_md, media_type="text/markdown")

@app.get("/jobs/{job_id}/claims")
async def get_job_claims(job_id: int, db: AsyncSession = Depends(get_db)):
    claims_res = await db.execute(
        select(Claim).where(Claim.job_id == job_id)
    )
    claims = claims_res.scalars().all()
    return [
        {
            "id": c.id,
            "claim": c.text,
            "quote": c.quote,
            "quote_found": c.quote_found,
            "entailment": c.entailment,
            "entailment_reason": c.entailment_reason,
            "final_status": c.final_status,
            "confidence": c.confidence,
            "shadow_verdict": c.shadow_verdict,
            "rewrite_count": c.rewrite_count
        }
        for c in claims
    ]

@app.get("/jobs/{job_id}/ledger")
async def get_job_ledger(job_id: int, db: AsyncSession = Depends(get_db)):
    ledger_res = await db.execute(
        select(LedgerEntry).where(LedgerEntry.job_id == job_id).order_by(LedgerEntry.id.asc())
    )
    entries = ledger_res.scalars().all()
    return [
        {
            "id": e.id,
            "phase": e.phase,
            "model": e.model,
            "tokens_in": e.tokens_in,
            "tokens_out": e.tokens_out,
            "cost_usd_modeled": float(e.cost_usd_modeled),
            "latency_ms": e.latency_ms,
            "cache_hit": e.cache_hit,
            "created_at": e.created_at
        }
        for e in entries
    ]

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job.status = "cancelled"
    await db.commit()
    return {"job_id": job.id, "status": "cancelled"}

@app.get("/export/claims.csv")
async def export_claims_csv(db: AsyncSession = Depends(get_db)):
    """Exports all claims to CSV for human labeling and evaluation."""
    claims_res = await db.execute(select(Claim, Job.question).join(Job, Claim.job_id == Job.id))
    rows = claims_res.all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "claim_id", "job_id", "question", "claim_text", "exact_quote",
        "quote_found", "entailment_verdict", "final_status", "human_label"
    ])

    for claim, question in rows:
        writer.writerow([
            claim.id, claim.job_id, question, claim.text, claim.quote,
            claim.quote_found, claim.entailment, claim.final_status, claim.human_label
        ])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=claims_evaluation.csv"}
    )