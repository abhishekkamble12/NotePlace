# app/research/nodes.py
"""
Node implementations for the CiteGuard LangGraph research StateGraph.
Every LLM call routes through app.core.llm (the single gate).
Checkpoints store only metadata; large evidence snapshots stay in Postgres.
"""
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.embeddings import embed_texts
from app.core.fetcher import chunk_spans, fetch_url_snapshot
from app.core.llm import PHASE_BUDGET_SHARES
from app.core.search import search_web
from app.core.verifier import DETERMINISTIC_DROP_CODES, rewrite_claim, verify_claim
from app.db import AsyncSessionLocal
from app.models import Chunk, Claim, Conflict, Job, LedgerEntry, Source, SubQuestion
from app.planner import generate_plan
from app.reporter import generate_markdown_report
from app.research.state import ClaimState, ConflictState, ResearchGraphState, SubQuestionState
from app.writer import draft_claims_for_subquestion


async def check_budget_ok(db: AsyncSession, job_id: int, phase: str, max_tokens: int) -> bool:
    """Checks whether current spend is within the allowed phase cap and job cap."""
    phase_share = PHASE_BUDGET_SHARES.get(phase, 0.25)
    phase_cap = int(max_tokens * phase_share)

    res = await db.execute(
        select(
            func.coalesce(
                func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out).filter(LedgerEntry.cache_hit.is_(False)),
                0
            ).label("total"),
            func.coalesce(
                func.sum(LedgerEntry.tokens_in + LedgerEntry.tokens_out).filter(
                    LedgerEntry.phase == phase,
                    LedgerEntry.cache_hit.is_(False)
                ),
                0
            ).label("phase_tokens"),
        ).where(LedgerEntry.scope_type == "job", LedgerEntry.scope_id == job_id)
    )
    row = res.one()
    total_spent = int(row.total)
    phase_spent = int(row.phase_tokens)

    if phase_spent >= phase_cap or total_spent >= max_tokens:
        return False
    return True


async def plan_node(state: ResearchGraphState) -> dict[str, Any]:
    """Decomposes the main question into 4-8 targeted sub-questions."""
    job_id = state["job_id"]
    question = state["question"]

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if job:
            job.stage = "planning"
            await db.commit()

        # If subquestions are already present in DB (e.g. resuming from checkpoint)
        res = await db.execute(select(SubQuestion).where(SubQuestion.job_id == job_id))
        existing_subqs = res.scalars().all()
        if existing_subqs:
            subqs_list: list[SubQuestionState] = [
                {"id": sq.id, "text": sq.text, "status": sq.status}
                for sq in existing_subqs
            ]
            return {"subquestions": subqs_list, "stage": "researching"}

        # Generate new plan
        created_subqs = await generate_plan(db, job_id, question)
        subqs_list = [
            {"id": sq.id, "text": sq.text, "status": sq.status}
            for sq in created_subqs
        ]
        return {"subquestions": subqs_list, "stage": "researching"}


async def research_node(state: ResearchGraphState) -> dict[str, Any]:
    """
    Researches sub-questions: web search -> SSRF-safe fetch -> snapshot store -> chunk -> embed.
    Honors the 50% research phase budget cap.
    """
    job_id = state["job_id"]
    subqs = state["subquestions"]
    max_tokens = state["max_tokens"]
    source_ids = list(state.get("source_ids", []))
    updated_subqs: list[SubQuestionState] = []

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if job:
            job.stage = "researching"
            await db.commit()

        for sq_state in subqs:
            sq_id = sq_state["id"]
            sq_text = sq_state["text"]

            if sq_state.get("status") == "done":
                updated_subqs.append(sq_state)
                continue

            # Check research budget cap before processing next sub-question
            budget_ok = await check_budget_ok(db, job_id, phase="research", max_tokens=max_tokens)
            if not budget_ok:
                # Mark remaining subquestions as skipped_budget per spec §6.5
                db_sq = await db.get(SubQuestion, sq_id)
                if db_sq:
                    db_sq.status = "skipped_budget"
                    await db.commit()
                updated_subqs.append({"id": sq_id, "text": sq_text, "status": "skipped_budget"})
                continue

            # 1. Search web for top N URLs
            urls = await search_web(sq_text, top_n=2)
            subq_sources = []

            # 2. Fetch snapshots & chunk & embed
            for url in urls:
                try:
                    domain, content_hash, snapshot = await fetch_url_snapshot(url)
                    # Check dedupe on (job_id, content_hash)
                    existing = await db.execute(
                        select(Source).where(Source.job_id == job_id, Source.content_hash == content_hash)
                    )
                    src = existing.scalar_one_or_none()
                    if not src:
                        src = Source(
                            job_id=job_id,
                            url=url,
                            domain=domain,
                            content_hash=content_hash,
                            snapshot=snapshot
                        )
                        db.add(src)
                        await db.commit()
                        await db.refresh(src)

                    if src.id not in source_ids:
                        source_ids.append(src.id)
                    subq_sources.append(src)

                    # Chunk and embed
                    # Chunks are stored RAW with snapshot offsets (text == snapshot[start:end]); the
                    # untrusted-data delimiters are added only when the prompt is built (writer.py).
                    spans = chunk_spans(snapshot, chunk_size=350, overlap=50)
                    if spans:
                        chunks_text = [snapshot[a:b] for a, b in spans]
                        embeddings = embed_texts(chunks_text)

                        for (a, b), body, emb in zip(spans, chunks_text, embeddings):
                            chk = Chunk(
                                source_id=src.id,
                                subq_id=sq_id,
                                text=body,
                                start_char=a,
                                end_char=b,
                                embedding=emb
                            )
                            db.add(chk)
                        await db.commit()

                except Exception as e:
                    # Individual fetch errors should not crash the job
                    print(f"Fetch skipped for {url}: {e}")
                    continue

            # Mark subquestion done
            db_sq = await db.get(SubQuestion, sq_id)
            if db_sq:
                db_sq.status = "done"
                await db.commit()
            updated_subqs.append({"id": sq_id, "text": sq_text, "status": "done"})

    return {
        "subquestions": updated_subqs,
        "source_ids": source_ids,
        "stage": "writing"
    }


async def write_node(state: ResearchGraphState) -> dict[str, Any]:
    """Drafts atomic claims supported by verbatim quotes for completed sub-questions."""
    job_id = state["job_id"]
    subqs = state["subquestions"]
    claims_list: list[ClaimState] = []

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if job:
            job.stage = "writing"
            await db.commit()

        # Check existing claims in case of resume
        existing_res = await db.execute(select(Claim).where(Claim.job_id == job_id))
        existing_claims = existing_res.scalars().all()
        if existing_claims:
            claims_list = [
                {
                    "id": c.id,
                    "subq_id": c.subq_id,
                    "text": c.text,
                    "source_id": c.source_id,
                    "quote": c.quote,
                    "quote_found": c.quote_found,
                    "entailment": c.entailment,
                    "entailment_reason": c.entailment_reason,
                    "final_status": c.final_status,
                    "rewrite_count": c.rewrite_count,
                    "confidence": c.confidence,
                    "shadow_verdict": c.shadow_verdict,
                }
                for c in existing_claims
            ]
            return {"claims": claims_list, "stage": "verifying"}

        for sq in subqs:
            if sq["status"] != "done":
                continue

            sq_id = sq["id"]
            sq_text = sq["text"]

            # Retrieve top chunks for this subquestion using plain SQLAlchemy query
            chunks_res = await db.execute(
                select(Chunk.id, Chunk.source_id, Chunk.start_char, Chunk.text)
                .where(Chunk.subq_id == sq_id, Chunk.start_char.is_not(None))
                .order_by(Chunk.id)
                .limit(4)
            )
            evidence_snippets = [
                {"chunk_id": cid, "source_id": sid, "start": st, "text": txt}
                for cid, sid, st, txt in chunks_res.all()
            ]

            if not evidence_snippets:
                continue

            created = await draft_claims_for_subquestion(
                db=db,
                job_id=job_id,
                subq_id=sq_id,
                subq_text=sq_text,
                evidence_snippets=evidence_snippets
            )

            for c in created:
                claims_list.append({
                    "id": c.id,
                    "subq_id": c.subq_id,
                    "text": c.text,
                    "source_id": c.source_id,
                    "quote": c.quote,
                    "quote_found": c.quote_found,
                    "entailment": c.entailment,
                    "entailment_reason": c.entailment_reason,
                    "final_status": c.final_status,
                    "rewrite_count": c.rewrite_count,
                    "confidence": c.confidence,
                    "shadow_verdict": c.shadow_verdict,
                })

    return {"claims": claims_list, "stage": "verifying"}


async def verify_node(state: ResearchGraphState) -> dict[str, Any]:
    """Runs Check 1 (Deterministic) and Check 2 (Cross-Model Entailment) on all claims."""
    job_id = state["job_id"]
    is_shadow = state.get("mode") == "shadow"
    updated_claims: list[ClaimState] = []

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if job:
            job.stage = "verifying"
            await db.commit()

        # Fetch claims with their snapshots
        res = await db.execute(
            select(Claim, Source.snapshot)
            .outerjoin(Source, Claim.source_id == Source.id)
            .where(Claim.job_id == job_id)
        )
        rows = res.all()

        for claim, snapshot in rows:
            snapshot_text = snapshot or ""
            # Only unverified claims are checked. Writer-rejected claims are stored as "dropped" and are never
            # sent to verification, in shadow mode too.
            if claim.final_status in ("pending", ""):
                v_res = await verify_claim(
                    db=db,
                    job_id=job_id,
                    claim_text=claim.text,
                    quote=claim.quote,
                    snapshot=snapshot_text,
                    start=claim.start_char,
                    end=claim.end_char,
                )

                claim.quote_found = v_res.quote_found
                claim.entailment = v_res.verdict
                claim.entailment_reason = (f"[{v_res.drop_reason}] " if v_res.drop_reason else "") + v_res.reason
                if v_res.span_start is not None and v_res.span_end is not None:
                    # The stored quote is always the snapshot span, never writer text.
                    claim.start_char, claim.end_char = v_res.span_start, v_res.span_end
                    claim.quote = snapshot_text[v_res.span_start:v_res.span_end]

                if is_shadow:
                    claim.shadow_verdict = v_res.verdict
                    # Stage-1 rejections are enforced; Stage-2 verdicts are recorded only.
                    claim.final_status = "dropped" if v_res.drop_reason in DETERMINISTIC_DROP_CODES else "kept"
                elif v_res.action == "kept":
                    claim.final_status = "kept"
                elif v_res.action == "needs_rewrite" and claim.rewrite_count < 1:
                    claim.final_status = "needs_rewrite"
                else:
                    # unsupported, quote/fact mismatch, invalid output: dropped, never rewritten
                    claim.final_status = "dropped"

            updated_claims.append({
                "id": claim.id,
                "subq_id": claim.subq_id,
                "text": claim.text,
                "source_id": claim.source_id,
                "quote": claim.quote,
                "quote_found": claim.quote_found,
                "entailment": claim.entailment,
                "entailment_reason": claim.entailment_reason,
                "final_status": claim.final_status,
                "rewrite_count": claim.rewrite_count,
                "confidence": claim.confidence,
                "shadow_verdict": claim.shadow_verdict,
            })

        await db.commit()

    return {"claims": updated_claims, "stage": "verifying"}


async def rewrite_node(state: ResearchGraphState) -> dict[str, Any]:
    """Performs a single precision rewrite attempt on claims flagged as needs_rewrite."""
    job_id = state["job_id"]
    updated_claims: list[ClaimState] = []

    async with AsyncSessionLocal() as db:
        for c_state in state["claims"]:
            if c_state["final_status"] == "needs_rewrite" and c_state["rewrite_count"] < 1:
                claim = await db.get(Claim, c_state["id"])
                if claim:
                    rewritten = await rewrite_claim(
                        db=db,
                        job_id=job_id,
                        original_claim=claim.text,
                        quote=claim.quote,
                        critique_reason=claim.entailment_reason
                    )
                    if rewritten:
                        claim.text = rewritten
                        claim.rewrite_count += 1
                        claim.final_status = "pending"
                    else:
                        claim.final_status = "dropped"
                    await db.commit()

                    updated_claims.append({
                        "id": claim.id,
                        "subq_id": claim.subq_id,
                        "text": claim.text,
                        "source_id": claim.source_id,
                        "quote": claim.quote,
                        "quote_found": claim.quote_found,
                        "entailment": claim.entailment,
                        "entailment_reason": claim.entailment_reason,
                        "final_status": claim.final_status,
                        "rewrite_count": claim.rewrite_count,
                        "confidence": claim.confidence,
                        "shadow_verdict": claim.shadow_verdict,
                    })
                    continue

            updated_claims.append(c_state)

    return {"claims": updated_claims}


async def analyze_node(state: ResearchGraphState) -> dict[str, Any]:
    """
    Detects cross-source factual contradictions and assigns multi-domain confidence.
    """
    job_id = state["job_id"]
    conflicts_list: list[ConflictState] = []

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if job:
            job.stage = "analyzing"
            await db.commit()

        # Fetch kept claims with source domain
        res = await db.execute(
            select(Claim, Source.domain)
            .outerjoin(Source, Claim.source_id == Source.id)
            .where(Claim.job_id == job_id, Claim.final_status == "kept")
        )
        kept_claims = res.all()

        # Group by sub-question to find domain diversity
        subq_domains = {}
        for c, domain in kept_claims:
            if c.subq_id not in subq_domains:
                subq_domains[c.subq_id] = set()
            if domain:
                subq_domains[c.subq_id].add(domain)

        # Set confidence per claim
        for c, domain in kept_claims:
            distinct_domains = len(subq_domains.get(c.subq_id, set()))
            if distinct_domains >= 2:
                c.confidence = "high"
            elif c.entailment == "supports":
                c.confidence = "medium"
            else:
                c.confidence = "low"

        # Conflict detection: pair-wise comparison of kept claims from different domains
        from rapidfuzz import fuzz
        opposing_pairs = [
            ("increase", "decrease"), ("reduce", "increase"), ("effective", "ineffective"),
            ("safe", "harmful"), ("risk", "benefit"), ("supports", "refutes"),
            ("significant", "negligible"), ("superior", "inferior"), ("positive", "negative")
        ]
        for i in range(len(kept_claims)):
            for j in range(i + 1, len(kept_claims)):
                c_a, domain_a = kept_claims[i]
                c_b, domain_b = kept_claims[j]
                if domain_a and domain_b and domain_a != domain_b:
                    sim = fuzz.token_set_ratio(c_a.text, c_b.text)
                    if sim >= 45:
                        txt_a_lower = c_a.text.lower()
                        txt_b_lower = c_b.text.lower()
                        has_contradiction = any(
                            (pos in txt_a_lower and neg in txt_b_lower) or (neg in txt_a_lower and pos in txt_b_lower)
                            for pos, neg in opposing_pairs
                        )
                        if has_contradiction:
                            note = f"Disagreement between {domain_a} and {domain_b} on findings."
                            conflict_record = Conflict(
                                job_id=job_id,
                                claim_a_id=c_a.id,
                                claim_b_id=c_b.id,
                                note=note
                            )
                            db.add(conflict_record)
                            conflicts_list.append({
                                "claim_a_id": c_a.id,
                                "claim_b_id": c_b.id,
                                "note": note
                            })

        await db.commit()

    return {"conflicts": conflicts_list, "stage": "rendering"}


async def render_node(state: ResearchGraphState) -> dict[str, Any]:
    """Renders the final 7-section verified Markdown report and updates job status."""
    job_id = state["job_id"]

    async with AsyncSessionLocal() as db:
        job = await db.get(Job, job_id)
        if not job:
            return {"status": "failed", "stage": "done"}

        report = await generate_markdown_report(db, job_id)
        job.report_md = report
        job.stage = "rendering"

        # If job was marked partial due to budget exit, preserve partial status
        if job.status != "partial" and job.status != "cancelled":
            job.status = "done"

        await db.commit()
        return {"status": job.status, "stage": "done"}
