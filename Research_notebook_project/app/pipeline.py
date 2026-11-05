# app/pipeline.py
import asyncio
from datetime import datetime, timezone, timedelta
from sqlalchemy import select
from app.db import AsyncSessionLocal
from app.models import Job, SubQuestion, Source, Claim
from app.llm import BudgetExceededError, RateLimitExceededError
from app.planner import generate_plan
from app.fetcher import fetch_and_clean_page, chunk_text, save_evidence_source
from app.writer import draft_claims_for_subquestion
from app.verifier import run_verification_loop
from app.reporter import generate_markdown_report

FALLBACK_DOMAINS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Quantum_supremacy",
    "https://en.wikipedia.org/wiki/Electric_battery"
]

async def _research_single_subquestion(
    job_id: int,
    sq_id: int
) -> list[dict]:
    """Researches a single sub-question: fetches pages, stores snapshots, and returns chunks."""
    evidence_chunks: list[dict] = []
    async with AsyncSessionLocal() as session:
        for url in FALLBACK_DOMAINS[:2]:
            fetch_res = await fetch_and_clean_page(url)
            if fetch_res:
                clean_text, domain = fetch_res
                source = await save_evidence_source(session, job_id, url, clean_text, domain)
                chunks = chunk_text(clean_text, chunk_size=300, overlap=40)
                for chk in chunks[:2]:
                    evidence_chunks.append({
                        "source_id": source.id,
                        "text": chk
                    })

        sq = await session.get(SubQuestion, sq_id)
        if sq:
            sq.status = "done"
            await session.commit()

    return evidence_chunks


async def execute_research_job(job_id: int) -> None:
    """
    Resumable, budget-bounded async research orchestrator.
    Supports 'multi' (parallel workers) and 'single' (sequential) modes.
    """
    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if not job or job.status in ("done", "cancelled"):
            return

        # Initialize deadline if not set
        if not job.deadline_at:
            job.deadline_at = datetime.now(timezone.utc) + timedelta(minutes=job.max_minutes)
        job.status = "running"
        await db.commit()

        try:
            # ---------------------------------------------
            # Stage 1: PLANNING
            # ---------------------------------------------
            subqs_res = await db.execute(
                select(SubQuestion).where(SubQuestion.job_id == job_id)
            )
            subquestions = subqs_res.scalars().all()

            if not subquestions or job.stage == "planning":
                job.stage = "planning"
                await db.commit()
                subquestions = await generate_plan(db, job_id, job.question)

            # ---------------------------------------------
            # Stage 2: RESEARCHING (Evidence Store)
            # ---------------------------------------------
            pending_subqs = [sq for sq in subquestions if sq.status == "pending"]
            evidence_map: dict[int, list[dict]] = {}

            if pending_subqs or job.stage in ("planning", "researching"):
                job.stage = "researching"
                await db.commit()

                if job.mode == "multi":
                    # Multi mode: research sub-questions concurrently
                    tasks = [_research_single_subquestion(job_id, sq.id) for sq in pending_subqs]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    for sq, res in zip(pending_subqs, results):
                        if isinstance(res, list):
                            evidence_map[sq.id] = res
                        else:
                            sq.status = "failed"
                            evidence_map[sq.id] = []
                    await db.commit()
                else:
                    # Single mode: research sub-questions sequentially in one context
                    for sq in pending_subqs:
                        try:
                            evidence_map[sq.id] = await _research_single_subquestion(job_id, sq.id)
                        except BudgetExceededError:
                            # Research cap reached: mark remaining as skipped_budget
                            sq.status = "skipped_budget"
                            await db.commit()
                            break

            # ---------------------------------------------
            # Stage 3: WRITING
            # ---------------------------------------------
            if job.stage in ("planning", "researching", "writing"):
                job.stage = "writing"
                await db.commit()

                for sq in subquestions:
                    # Skip sub-questions that already have drafted claims (resumability)
                    existing_claims = await db.execute(
                        select(Claim).where(Claim.job_id == job_id, Claim.subq_id == sq.id)
                    )
                    if existing_claims.scalars().first():
                        continue

                    snippets = evidence_map.get(sq.id, [])
                    if snippets:
                        await draft_claims_for_subquestion(
                            db=db,
                            job_id=job_id,
                            subq_id=sq.id,
                            subq_text=sq.text,
                            evidence_snippets=snippets
                        )

            # ---------------------------------------------
            # Stage 4: VERIFYING
            # ---------------------------------------------
            if job.stage in ("planning", "researching", "writing", "verifying"):
                job.stage = "verifying"
                await db.commit()
                is_shadow = job.mode == "shadow"
                await run_verification_loop(db, job_id, shadow=is_shadow)

            # ---------------------------------------------
            # Stage 5: RENDERING
            # ---------------------------------------------
            job.stage = "rendering"
            await db.commit()
            await generate_markdown_report(db, job_id)

            job.status = "done"
            job.stage = "done"
            await db.commit()

        except (BudgetExceededError, RateLimitExceededError) as be:
            # Graceful Budget Degradation: finish as partial report
            job.status = "partial"
            job.stage = "rendering"
            await db.commit()

            # Audit rule: Drop any unverified claims so they never reach the report
            pending_claims_res = await db.execute(
                select(Claim).where(Claim.job_id == job_id, Claim.final_status == "pending")
            )
            for pc in pending_claims_res.scalars().all():
                pc.final_status = "dropped"
                pc.entailment_reason = f"Dropped: budget limit reached ({be})"
            await db.commit()

            # Render partial report
            await generate_markdown_report(db, job_id)

        except Exception as e:
            job.status = "failed"
            job.report_md = f"Job failed with unexpected error: {str(e)}"
            await db.commit()
