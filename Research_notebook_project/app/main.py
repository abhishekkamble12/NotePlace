# app/main.py
import csv
import hmac
import io
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import get_current_user
from app.chat import answer as answer_question
from app.core.accounting import ledger_summary
from app.core.verifier import require_evidence_key
from app.db import get_db, init_db
from app.ingest.service import PdfError, create_notebook, ingest_pdf
from app.models import Chunk, Claim, Conversation, Job, LedgerEntry, Message, Notebook, Source, User
from app.pipeline import execute_research_job


def require_admin(x_api_key: str | None = Header(default=None)):
    """Fail closed: with ADMIN_API_KEY unset, the admin endpoints are disabled (403), never open."""
    expected = os.environ.get("ADMIN_API_KEY", "")
    if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
        raise HTTPException(status_code=403, detail="Forbidden")


def _csv_safe(v):
    """Neutralize spreadsheet formula injection (OWASP CSV injection)."""
    v = "" if v is None else v
    return "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs when server starts: ensures pgvector & database setup
    require_evidence_key()
    await init_db()
    yield

app = FastAPI(
    title="CiteGuard API",
    description="Budget-Bounded Research Service with Verified Citations",
    version="0.2.0",
    lifespan=lifespan
)


class ResearchRequest(BaseModel):
    question: str = Field(min_length=5, max_length=2000, description="The research question to investigate")
    max_tokens: int = Field(default=50_000, ge=1_000, le=500_000, description="Hard token budget cap")
    max_minutes: int = Field(default=8, ge=1, le=60, description="Time deadline in minutes")
    mode: Literal["standard", "shadow"] = Field(
        default="standard",
        description="standard drops claims the verifier rejects; shadow records the verdicts but keeps every claim"
    )

@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    result = await db.execute(text("SELECT 1"))
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

    # Aggregate spend from ledger using scope_type and scope_id
    usage = await ledger_summary(db, "job", job_id)

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
        "tokens_spent": usage["budget_accounted_tokens"],  # budget-accounted (reported + retained reservations), not provider-reported
        "tokens_budget": job.max_tokens,
        "usd_spent": usage["modeled_usd_reported"] + usage["modeled_usd_legacy"],  # modeled at list price
        "usage": usage,
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
        select(LedgerEntry)
        .where(LedgerEntry.scope_type == "job", LedgerEntry.scope_id == job_id)
        .order_by(LedgerEntry.id.asc())
    )
    entries = ledger_res.scalars().all()
    return [
        {
            "id": e.id,
            "phase": e.phase,
            "kind": e.kind,
            "model": e.model,
            "tokens_in": e.tokens_in,
            "tokens_out": e.tokens_out,
            "cost_usd_modeled": float(e.cost_usd_modeled),
            "latency_ms": e.latency_ms,
            "cache_hit": e.cache_hit,
            "usage_provenance": e.usage_provenance,  # reported | unknown | legacy; tokens_* are 0 when unknown
            "usage_known": e.usage_known,
            "status": e.status,
            "provider": e.provider,
            "call_id": e.call_id,
            "reserved_tokens": e.reserved_tokens,
            "est_tokens_in": e.est_tokens_in,
            "created_at": e.created_at
        }
        for e in entries
    ]

@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job.status in ("queued", "running"):  # never rewrite a finished job
        job.status = "cancelled"
        await db.commit()
    return {"job_id": job.id, "status": job.status}

# ---------------------------------------------------------------------------
# Notebooks, PDF sources and grounded chat. Every query filters by the caller's user id in SQL; another user's
# notebook or conversation is a 404, so ids do not reveal what exists.
# ---------------------------------------------------------------------------
MAX_PDF_BYTES = 20 * 1024 * 1024


class NotebookIn(BaseModel):
    title: str = Field(default="", max_length=255)


class ChatIn(BaseModel):
    question: str = Field(min_length=2, max_length=2000)


async def _notebook(db: AsyncSession, user: User, notebook_id: int) -> Notebook:
    nb = (await db.execute(select(Notebook).where(Notebook.id == notebook_id, Notebook.user_id == user.id))).scalar_one_or_none()
    if nb is None:
        raise HTTPException(status_code=404, detail="Notebook not found")
    return nb


async def _conversation(db: AsyncSession, user: User, conversation_id: int) -> Conversation:
    c = (await db.execute(select(Conversation).where(Conversation.id == conversation_id, Conversation.user_id == user.id))).scalar_one_or_none()
    if c is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return c


def _message_out(m: Message) -> dict:
    return {"id": m.id, "role": m.role, "content": m.content, "standalone_question": m.standalone_question,
            "citations": m.citations or [], "dropped": m.dropped or [], "abstained": m.abstained,
            "abstain_reason": m.abstain_reason, "created_at": m.created_at}


@app.post("/notebooks", status_code=201)
async def create_notebook_ep(body: NotebookIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    nb = await create_notebook(db, user.id, body.title)
    await db.commit()
    return {"id": nb.id, "title": nb.title}


@app.get("/notebooks")
async def list_notebooks(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(Notebook).where(Notebook.user_id == user.id).order_by(Notebook.id))).scalars().all()
    return [{"id": n.id, "title": n.title} for n in rows]


@app.post("/notebooks/{notebook_id}/sources", status_code=201)
async def upload_pdf(notebook_id: int, file: UploadFile = File(...), user: User = Depends(get_current_user),
                     db: AsyncSession = Depends(get_db)):
    await _notebook(db, user, notebook_id)
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail=f"PDF larger than {MAX_PDF_BYTES // (1024 * 1024)} MB")
    if not data.startswith(b"%PDF-"):  # magic bytes, not the client-supplied content type
        raise HTTPException(status_code=415, detail="Only PDF files are accepted")
    try:
        # shortcut: parsing + embedding run inside the request (CPU-bound, blocks this worker); move to the SQS worker in Week 4
        src, created = await ingest_pdf(db, user.id, notebook_id, data, filename=(file.filename or "")[:200])
    except PdfError as e:
        raise HTTPException(status_code=422, detail=f"Unreadable PDF: {e}")
    n = (await db.execute(select(func.count(Chunk.id)).where(Chunk.source_id == src.id))).scalar_one()
    await db.commit()
    return {"source_id": src.id, "title": src.title, "created": created, "chunks": n}


@app.get("/notebooks/{notebook_id}/sources")
async def list_sources(notebook_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _notebook(db, user, notebook_id)
    rows = (await db.execute(select(Source).where(Source.notebook_id == notebook_id).order_by(Source.id))).scalars().all()
    return [{"id": s.id, "title": s.title, "pages": len(s.page_offsets or []), "fetched_at": s.fetched_at} for s in rows]


@app.post("/notebooks/{notebook_id}/conversations", status_code=201)
async def create_conversation(notebook_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _notebook(db, user, notebook_id)
    c = Conversation(notebook_id=notebook_id, user_id=user.id)
    db.add(c)
    await db.commit()
    return {"id": c.id, "notebook_id": notebook_id, "max_tokens": c.max_tokens}


@app.post("/conversations/{conversation_id}/messages")
async def chat(conversation_id: int, body: ChatIn, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    conv = await _conversation(db, user, conversation_id)
    return _message_out(await answer_question(db, user.id, conv, body.question))


@app.get("/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _conversation(db, user, conversation_id)
    rows = (await db.execute(select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id))).scalars().all()
    return [_message_out(m) for m in rows]


@app.get("/conversations/{conversation_id}/ledger")
async def conversation_ledger(conversation_id: int, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    await _conversation(db, user, conversation_id)
    return await ledger_summary(db, "conversation", conversation_id)


@app.get("/export/claims.csv", dependencies=[Depends(require_admin)])
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
        writer.writerow([_csv_safe(c) for c in (
            claim.id, claim.job_id, question, claim.text, claim.quote,
            claim.quote_found, claim.entailment, claim.final_status, claim.human_label
        )])

    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=claims_evaluation.csv"}
    )

@app.post("/labels", dependencies=[Depends(require_admin)])
async def import_labels_csv(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Imports human labels from CSV to evaluate verifier metrics."""
    contents = await file.read()
    try:
        rows = list(csv.DictReader(io.StringIO(contents.decode("utf-8"))))
        ids = [int(r["claim_id"]) for r in rows]
    except (UnicodeDecodeError, ValueError, KeyError, TypeError, csv.Error):
        raise HTTPException(status_code=400, detail="Malformed labels CSV")
    updated_count = 0
    for row, claim_id in zip(rows, ids):
        label = row.get("human_label", "").strip()
        if label:
            claim = await db.get(Claim, claim_id)
            if claim:
                claim.human_label = label
                updated_count += 1
    await db.commit()
    return {"status": "ok", "updated_claims": updated_count}
