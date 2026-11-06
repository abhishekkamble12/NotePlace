# app/mcp/server.py
"""
CiteGuard Official Model Context Protocol (MCP) Server (Week 4 / Spec §D23).
Exposes the verified citation research engine to MCP clients (Claude Desktop, Cursor, Antigravity).

Tools:
- research(question, max_tokens=60000, max_minutes=8, mode="multi"):
    Decomposes questions, gathers evidence, drafts claims, verifies citations,
    and returns a verified Markdown report and claim metadata.
- verify_quote(claim, quote, context=""):
    Directly runs CiteGuard's 2-stage verifier (exact/fuzzy snapshot match + entailment).
"""
import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from app.core.accounting import ledger_summary
from app.core.verifier import require_evidence_key, verify_claim
from app.db import AsyncSessionLocal, init_db
from app.models import Claim, Job
from app.pipeline import execute_research_job
from mcp.server.mcpserver import MCPServer

mcp_server = MCPServer(name="citeguard", version="1.0.0")


class ResearchInput(BaseModel):
    question: str = Field(description="The research question to investigate with citation verification.")
    max_tokens: int = Field(default=60000, description="Hard token budget cap for the research job.")
    max_minutes: int = Field(default=8, description="Maximum wall-clock execution time in minutes.")
    mode: str = Field(default="multi", description="'multi' (parallel sub-questions) or 'single' (sequential).")


@mcp_server.tool()
async def research(
    question: str,
    max_tokens: int = 60000,
    max_minutes: int = 8,
    mode: str = "multi"
) -> str:
    """
    Executes a budget-bounded, citation-verified research run on any question.
    Returns the 7-section verified Markdown report and structured audit metadata.
    """
    await init_db()

    # Create job in database
    async with AsyncSessionLocal() as db:
        job = Job(
            question=question,
            max_tokens=max_tokens,
            max_minutes=max_minutes,
            mode=mode,
            status="pending",
            stage="planning",
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        job_id = job.id

    # Execute the research graph pipeline
    result_state = await execute_research_job(job_id)

    # Fetch final report and claims from database
    async with AsyncSessionLocal() as db:
        final_job = await db.get(Job, job_id)
        report_md = final_job.report_md if final_job else "# Research Incomplete"

        # Fetch claims
        from sqlalchemy import select
        claims_res = await db.execute(select(Claim).where(Claim.job_id == job_id))
        claims = claims_res.scalars().all()

        # Fetch ledger stats
        usage = await ledger_summary(db, "job", job_id)
        total_tokens = usage["budget_accounted_tokens"]
        total_cost = usage["modeled_usd_reported"] + usage["modeled_usd_legacy"]

    payload = {
        "job_id": job_id,
        "status": final_job.status if final_job else result_state.get("status"),
        "total_tokens_used": total_tokens,  # budget-accounted; see "usage" for the reported/unknown split
        "usage": usage,
        "modeled_usd": round(total_cost, 4),
        "total_claims": len(claims),
        "kept_claims": sum(1 for c in claims if c.final_status == "kept"),
        "dropped_claims": sum(1 for c in claims if c.final_status == "dropped"),
        "report_markdown": report_md,
    }

    return json.dumps(payload, indent=2)


# Standalone verify_quote runs get their own short-lived Job so every provider call is ledgered and budget-capped.
VERIFY_QUOTE_MAX_TOKENS = 8_000


@mcp_server.tool()
async def verify_quote(
    claim: str,
    quote: str,
    context: str = ""
) -> str:
    """
    Assesses whether a caller-supplied quote (and optional caller-supplied context) supports a claim,
    using CiteGuard's verifier (number/date/entity check + entailment).
    NOT source-authenticated: the text is not checked against any stored source snapshot.
    """
    await init_db()
    span_source = "caller_supplied_context" if context else "caller_supplied_quotation"
    async with AsyncSessionLocal() as db:
        job = Job(
            question=f"verify_quote: {claim[:200]}",
            max_tokens=VERIFY_QUOTE_MAX_TOKENS,
            max_minutes=2,
            mode="single",
            status="running",
            stage="verifying",
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        job_id = job.id

        res = await verify_claim(
            db=db,
            job_id=job_id,
            claim_text=claim,
            quote=quote,
            snapshot=context or quote,
            verifier_model="qwen/qwen3.8-27b",
        )

        job.status = "done"
        job.stage = "done"
        await db.commit()

    out = {
        "job_id": job_id,
        "quote_found": res.quote_found,
        "match_score": res.match_score,
        "entailment_verdict": res.verdict,
        "action": res.action,
        "reason": res.reason,
        "drop_reason": res.drop_reason,
        "source_authenticated": False,
        "span_source": span_source,
    }
    return json.dumps(out, indent=2)


def main():
    """Runs the MCP server via stdio for MCP clients."""
    require_evidence_key()
    mcp_server.run()


if __name__ == "__main__":
    main()
