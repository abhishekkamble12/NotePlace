# app/models.py
from datetime import datetime
from sqlalchemy import (
    String, Text, Integer, Numeric, ForeignKey,
    DateTime, Boolean, UniqueConstraint, func
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    max_tokens: Mapped[int] = mapped_column(Integer, default=50_000)
    max_usd: Mapped[float] = mapped_column(Numeric(6, 2), default=0.50)
    max_minutes: Mapped[int] = mapped_column(Integer, default=8)
    mode: Mapped[str] = mapped_column(String(10), default="multi")   # "single" | "multi"
    status: Mapped[str] = mapped_column(String(20), default="queued") # "queued" | "done" | "partial" | "failed" | "cancelled"
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
    subq_id: Mapped[int] = mapped_column(ForeignKey("subquestions.id"))
    text: Mapped[str] = mapped_column(Text)
    embedding = mapped_column(Vector(384))                            # 384 dimensions for bge-small

class Claim(Base):
    __tablename__ = "claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    subq_id: Mapped[int | None] = mapped_column(ForeignKey("subquestions.id"))
    text: Mapped[str] = mapped_column(Text)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("sources.id"))
    quote: Mapped[str] = mapped_column(Text)
    quote_found: Mapped[bool | None] = mapped_column(Boolean)
    entailment: Mapped[str] = mapped_column(String(15), default="")         # "supports" | "partial" | "unsupported"
    entailment_reason: Mapped[str] = mapped_column(Text, default="")
    final_status: Mapped[str] = mapped_column(String(15), default="pending")# "kept" | "rewritten" | "dropped"
    rewrite_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[str] = mapped_column(String(10), default="low")      # "high" | "medium" | "low"
    shadow_verdict: Mapped[str] = mapped_column(String(15), default="")    # Recorded verdict during shadow mode
    human_label: Mapped[str] = mapped_column(String(15), default="")       # For evaluation benchmark

class Conflict(Base):
    __tablename__ = "conflicts"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    claim_a_id: Mapped[int] = mapped_column(ForeignKey("claims.id"))
    claim_b_id: Mapped[int] = mapped_column(ForeignKey("claims.id"))
    note: Mapped[str] = mapped_column(Text)

class LedgerEntry(Base):
    __tablename__ = "ledger"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"))
    phase: Mapped[str] = mapped_column(String(20))                         # "planning" | "research" | "writing" | "verifying"
    kind: Mapped[str] = mapped_column(String(20), default="llm")          # "llm" | "search" | "fetch"
    model: Mapped[str] = mapped_column(String(50), default="")
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd_modeled: Mapped[float] = mapped_column(Numeric(8, 5), default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

class VerifierTestSet(Base):
    """Week 1 benchmark test set (~150 claim/quote pairs)."""
    __tablename__ = "verifier_testset"

    id: Mapped[int] = mapped_column(primary_key=True)
    quote: Mapped[str] = mapped_column(Text)
    claim: Mapped[str] = mapped_column(Text)
    context: Mapped[str] = mapped_column(Text, default="")
    perturbation_type: Mapped[str] = mapped_column(String(50)) # "original" | "date_changed" | "overgeneralized" | "causal"
    label: Mapped[str] = mapped_column(String(15))             # "supports" | "partial" | "unsupported"