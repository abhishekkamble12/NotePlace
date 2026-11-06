# app/pipeline.py
"""
Entry point for running CiteGuard research jobs via the LangGraph StateGraph.
Threaded with job_id as the graph thread_id (in-memory checkpointer; not crash-safe).
"""
import logging
import traceback
from datetime import UTC, datetime, timedelta

from app.db import AsyncSessionLocal
from app.models import Job
from app.research.graph import research_graph
from app.research.state import ResearchGraphState

logger = logging.getLogger(__name__)


async def execute_research_job(job_id: int) -> None:
    """
    Executes a research job by invoking the LangGraph StateGraph.
    thread_id = str(job_id).
    """
    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if not job or job.status in ("done", "cancelled"):
            return

        if not job.deadline_at:
            job.deadline_at = datetime.now(UTC) + timedelta(minutes=job.max_minutes)
        job.status = "running"
        await db.commit()

        initial_state: ResearchGraphState = {
            "job_id": job.id,
            "question": job.question,
            "mode": job.mode, # type: ignore
            "stage": job.stage,
            "status": job.status,
            "max_tokens": job.max_tokens,
            "max_minutes": job.max_minutes,
            "subquestions": [],
            "source_ids": [],
            "claims": [],
            "conflicts": [],
            "error": None,
        }

    try:
        # Invoke LangGraph StateGraph with thread_id checkpointing
        config = {"configurable": {"thread_id": str(job_id)}}
        await research_graph.ainvoke(initial_state, config=config)

    except Exception as e:
        async with AsyncSessionLocal() as db:
            failed_job = await db.get(Job, job_id)
            if failed_job and failed_job.status != "cancelled":
                failed_job.status = "failed"
                failed_job.report_md = f"# Research Failed\n\nThe job failed ({type(e).__name__}). See server logs."
                logger.error("job %s failed:\n%s", job_id, traceback.format_exc())
                await db.commit()
