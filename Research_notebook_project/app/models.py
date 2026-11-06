# app/models.py
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    true,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    max_tokens: Mapped[int] = mapped_column(Integer, default=50_000)
    max_minutes: Mapped[int] = mapped_column(Integer, default=8)
    mode: Mapped[str] = mapped_column(String(10), default="multi")     # "single" | "multi" | "shadow"
    status: Mapped[str] = mapped_column(String(20), default="queued")  # "queued" | "done" | "partial" | "failed" | "cancelled"
    stage: Mapped[str] = mapped_column(String(20), default="planning") # "planning" | "researching" | "writing" | "verifying" | "rendering"
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    report_md: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class SubQuestion(Base):
    __tablename__ = "subquestions"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    text: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="pending") # "pending" | "done" | "failed" | "skipped_budget"

class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (UniqueConstraint("job_id", "content_hash"),)   # Deduplication idempotency

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    url: Mapped[str] = mapped_column(String(1000))
    domain: Mapped[str] = mapped_column(String(255), default="")      # For multi-domain independence checks
    content_hash: Mapped[str] = mapped_column(String(64))             # SHA-256
    snapshot: Mapped[str] = mapped_column(Text)                       # Full cleaned text
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    subq_id: Mapped[int | None] = mapped_column(ForeignKey("subquestions.id"), nullable=True)
    text: Mapped[str] = mapped_column(Text)                          # raw snapshot[start_char:end_char]
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding = mapped_column(Vector(384))                            # 384 dimensions for bge-small

class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    subq_id: Mapped[int | None] = mapped_column(ForeignKey("subquestions.id"), nullable=True)
    text: Mapped[str] = mapped_column(Text)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"), nullable=True)
    quote: Mapped[str] = mapped_column(Text)                          # derived from the snapshot span, never writer text
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("chunks.id"), nullable=True)
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)  # citation by reference: snapshot offsets
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quote_found: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    entailment: Mapped[str] = mapped_column(String(15), default="")          # "supports" | "partial" | "unsupported"
    entailment_reason: Mapped[str] = mapped_column(Text, default="")
    final_status: Mapped[str] = mapped_column(String(15), default="pending") # "kept" | "rewritten" | "dropped"
    rewrite_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[str] = mapped_column(String(10), default="low")       # "high" | "medium" | "low"
    shadow_verdict: Mapped[str] = mapped_column(String(15), default="")     # Recorded verdict during shadow mode
    human_label: Mapped[str] = mapped_column(String(15), default="")        # For evaluation benchmark

class Conflict(Base):
    __tablename__ = "conflicts"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    claim_a_id: Mapped[int] = mapped_column(ForeignKey("claims.id"))
    claim_b_id: Mapped[int] = mapped_column(ForeignKey("claims.id"))
    note: Mapped[str] = mapped_column(Text, default="")

class LedgerEntry(Base):
    """
    Cost, latency, and token ledger.
    scope_type + scope_id supports ("job", job_id) for Phase 1 and ("notebook", notebook_id) for Phase 2.
    """
    __tablename__ = "ledger"
    __table_args__ = (
        Index("ix_ledger_scope", "scope_type", "scope_id"),
        # provenance 'unknown' <=> usage_known is false; 'reported' and 'legacy' rows carry counted tokens
        CheckConstraint("(usage_provenance = 'unknown') = (NOT usage_known)", name="ck_ledger_provenance_known"),
        CheckConstraint("usage_provenance IN ('reported', 'unknown', 'legacy')", name="ck_ledger_provenance_values"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    scope_type: Mapped[str] = mapped_column(String(20), default="job", index=True) # "job" | "notebook"
    scope_id: Mapped[int] = mapped_column(Integer, index=True)
    phase: Mapped[str] = mapped_column(String(20))                                 # "planning" | "research" | "writing" | "verifying" | "reserve"
    kind: Mapped[str] = mapped_column(String(20), default="llm")                   # "llm" | "search" | "fetch" | "embed"
    model: Mapped[str] = mapped_column(String(50), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd_modeled: Mapped[float] = mapped_column(Numeric(8, 5), default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    # Usage accounting. tokens_in/out are PROVIDER-REPORTED only (0 when usage_known is False).
    # reserved_tokens is the pre-call reservation; est_tokens_in is the local estimator. Neither is actual usage.
    # reported = provider usage fields; unknown = no usage reported (budget charges reserved_tokens);
    # legacy = row written before migration 003: tokens may be provider-reported OR a local estimate, unverifiable.
    usage_provenance: Mapped[str] = mapped_column(String(10), default="reported")
    usage_known: Mapped[bool] = mapped_column(Boolean, default=True, server_default=true())
    reserved_tokens: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    est_tokens_in: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    status: Mapped[str] = mapped_column(String(10), default="ok", server_default="ok")  # "reserved" (in flight / crashed) | "ok" | "failed"
    provider: Mapped[str] = mapped_column(String(30), default="", server_default="")
    call_id: Mapped[str] = mapped_column(String(36), default="", server_default="")  # groups attempts of one call_llm call
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    @property
    def job_id(self) -> int | None:
        return self.scope_id if self.scope_type == "job" else None

class VerifierTestSet(Base):
    """Week 1 benchmark test set (~150 claim/quote pairs)."""
    __tablename__ = "verifier_testset"

    id: Mapped[int] = mapped_column(primary_key=True)
    quote: Mapped[str] = mapped_column(Text)
    claim: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text, default="")
    perturbation_type: Mapped[str] = mapped_column(String(50)) # "original" | "number" | "overgeneralize" | "entity" | "causal"
    label: Mapped[str] = mapped_column(String(15))             # "supports" | "partial" | "unsupported"
    source: Mapped[str] = mapped_column(String(500), default="")