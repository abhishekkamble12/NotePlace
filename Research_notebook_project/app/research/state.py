# app/research/state.py
"""
Typed graph state for CiteGuard research pipeline.
Rule: Large data blobs (snapshots, raw chunks) stay in PostgreSQL;
the state holds IDs and lightweight metadata to keep LangGraph checkpoints compact.
"""
from typing import Literal, TypedDict


class SubQuestionState(TypedDict):
    id: int
    text: str
    status: str  # "pending" | "done" | "failed" | "skipped_budget"

class ClaimState(TypedDict):
    id: int
    subq_id: int | None
    text: str
    source_id: int | None
    quote: str
    quote_found: bool | None
    entailment: str
    entailment_reason: str
    final_status: str  # "pending" | "kept" | "needs_rewrite" | "dropped"
    rewrite_count: int
    confidence: str    # "high" | "medium" | "low"
    shadow_verdict: str

class ConflictState(TypedDict):
    claim_a_id: int
    claim_b_id: int
    note: str

class ResearchGraphState(TypedDict):
    job_id: int
    question: str
    mode: Literal["standard", "shadow"]
    stage: str
    status: str
    max_tokens: int
    max_minutes: int
    subquestions: list[SubQuestionState]
    source_ids: list[int]
    claims: list[ClaimState]
    conflicts: list[ConflictState]
    error: str | None
