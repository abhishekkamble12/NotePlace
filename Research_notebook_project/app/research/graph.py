# app/research/graph.py
"""
LangGraph StateGraph definition for the CiteGuard Research Service.
Orchestrates:
  plan -> research -> write -> verify <-> rewrite -> analyze -> render -> END
Budget/cancel check runs on the plan, research and write edges. MemorySaver checkpointer: not crash-safe.
"""
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from sqlalchemy import func, select

from app.db import AsyncSessionLocal
from app.models import Job, LedgerEntry
from app.research.nodes import (
    analyze_node,
    plan_node,
    render_node,
    research_node,
    rewrite_node,
    verify_node,
    write_node,
)
from app.research.state import ResearchGraphState


async def check_budget_or_cancelled(state: ResearchGraphState) -> Literal["proceed", "over_budget"]:
    """
    Conditional edge evaluator placed before pipeline stages:
    Checks if job status is cancelled or if hard token budget ceiling has been reached.
    If over budget, diverts directly to render_node to produce a partial report.
    """
    job_id = state["job_id"]
    max_tokens = state["max_tokens"]

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if not job or job.status == "cancelled":
            return "over_budget"

        # Check total tokens consumed so far
        res = await db.execute(
            select(
                func.coalesce(
                    func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out).filter(LedgerEntry.cache_hit.is_(False)),
                    0
                )
            ).where(LedgerEntry.scope_type == "job", LedgerEntry.scope_id == job_id)
        )
        total_tokens = int(res.scalar() or 0)

        if total_tokens >= max_tokens:
            job.status = "partial"
            await db.commit()
            return "over_budget"

    return "proceed"


def check_rewrite_needed(state: ResearchGraphState) -> Literal["rewrite", "analyze"]:
    """
    Conditional edge after verify_node:
    If any claims were flagged as needs_rewrite and have not exceeded 1 rewrite attempt,
    routes to rewrite_node. Otherwise proceeds to cross-domain conflict analysis.
    """
    claims = state.get("claims", [])
    needs_rewrite = any(
        c.get("final_status") == "needs_rewrite" and c.get("rewrite_count", 0) < 1
        for c in claims
    )
    if needs_rewrite:
        return "rewrite"
    return "analyze"


def build_research_graph(checkpointer=None):
    """Constructs and compiles the CiteGuard research StateGraph."""
    workflow = StateGraph(ResearchGraphState)

    # 1. Register nodes
    workflow.add_node("plan", plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("write", write_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("analyze", analyze_node)
    workflow.add_node("render", render_node)

    # 2. Add edges with budget_check guards
    workflow.add_edge(START, "plan")

    workflow.add_conditional_edges(
        "plan",
        check_budget_or_cancelled,
        {
            "proceed": "research",
            "over_budget": "render",
        }
    )

    workflow.add_conditional_edges(
        "research",
        check_budget_or_cancelled,
        {
            "proceed": "write",
            "over_budget": "render",
        }
    )

    workflow.add_conditional_edges(
        "write",
        check_budget_or_cancelled,
        {
            "proceed": "verify",
            "over_budget": "render",
        }
    )

    workflow.add_conditional_edges(
        "verify",
        check_rewrite_needed,
        {
            "rewrite": "rewrite",
            "analyze": "analyze",
        }
    )

    # Bounded rewrite loop: rewrite routes back to verify for re-checking
    workflow.add_edge("rewrite", "verify")

    workflow.add_edge("analyze", "render")
    workflow.add_edge("render", END)

    if checkpointer is None:
        checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)


# Global compiled graph instance with memory checkpointing
research_graph = build_research_graph()
