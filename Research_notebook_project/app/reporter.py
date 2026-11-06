# app/reporter.py
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.accounting import ledger_summary
from app.models import Claim, Conflict, Job, Source, SubQuestion


async def generate_markdown_report(db: AsyncSession, job_id: int) -> str:
    """
    Renders the final structured Markdown report following CiteGuard specification:
    1. Executive Summary
    2. Findings per Sub-Question (numbered citations)
    3. Where Sources Disagree
    4. Evidence Gaps & Low-Confidence Claims
    5. Dropped Claims (audit transparency)
    6. References (URL, fetched_at, quote)
    7. Run Metadata (models, tokens, cost, latency)
    """
    job = await db.get(Job, job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    # Fetch subquestions
    subqs_res = await db.execute(
        select(SubQuestion).where(SubQuestion.job_id == job_id)
    )
    subquestions = subqs_res.scalars().all()

    # Fetch all claims
    claims_res = await db.execute(
        select(Claim).where(Claim.job_id == job_id)
    )
    all_claims = claims_res.scalars().all()

    # Fetch sources
    sources_res = await db.execute(
        select(Source).where(Source.job_id == job_id)
    )
    sources = {s.id: s for s in sources_res.scalars().all()}

    # Fetch ledger summary
    usage = await ledger_summary(db, "job", job_id)

    kept_claims = [c for c in all_claims if c.final_status == "kept"]
    dropped_claims = [c for c in all_claims if c.final_status == "dropped"]

    # Number references
    citation_map = {} # source_id -> citation index [1], [2]
    citations_list = []
    for c in kept_claims:
        if c.source_id and c.source_id not in citation_map:
            idx = len(citations_list) + 1
            citation_map[c.source_id] = idx
            citations_list.append((idx, sources.get(c.source_id), c.quote))

    report_lines = []
    report_lines.append(f"# Research Report: {job.question}\n")
    report_lines.append(f"> **Status:** `{job.status}` | **Execution Mode:** `{job.mode}` | **Stage:** `{job.stage}`\n")

    # 1. Executive Summary
    report_lines.append("## 1. Executive Summary\n")
    if kept_claims:
        report_lines.append(f"This report presents research synthesized across {len(subquestions)} sub-questions. A total of {len(all_claims)} draft claims were investigated, resulting in **{len(kept_claims)} verified claims** backed by exact quotes and **{len(dropped_claims)} unverified or dropped claims**.")
    else:
        report_lines.append("No claims met the verification criteria.")
    report_lines.append("\n")

    # 2. Findings per Sub-Question
    report_lines.append("## 2. Findings by Research Question\n")
    for sq in subquestions:
        report_lines.append(f"### {sq.text}\n")
        sq_claims = [c for c in kept_claims if c.subq_id == sq.id]
        if sq_claims:
            for c in sq_claims:
                cite_idx = citation_map.get(c.source_id, "?")
                confidence_tag = f"*(Confidence: {c.confidence})*"
                report_lines.append(f"- {c.text} [{cite_idx}] {confidence_tag}")
        else:
            report_lines.append("*No verified claims survived verification for this sub-question.*")
        report_lines.append("")

    # 3. Where Sources Disagree
    report_lines.append("## 3. Where Sources Disagree\n")
    conflicts_res = await db.execute(
        select(Conflict).where(Conflict.job_id == job_id)
    )
    conflicts = conflicts_res.scalars().all()
    if conflicts:
        report_lines.append("| Source A Claim | Source B Claim | Disagreement Analysis |")
        report_lines.append("|---|---|---|")
        for conf in conflicts:
            c_a = next((c for c in kept_claims if c.id == conf.claim_a_id), None)
            c_b = next((c for c in kept_claims if c.id == conf.claim_b_id), None)
            txt_a = (c_a.text if c_a else f"Claim #{conf.claim_a_id}").replace("|", "\\|")
            txt_b = (c_b.text if c_b else f"Claim #{conf.claim_b_id}").replace("|", "\\|")
            note = (conf.note or "Contradictory findings detected").replace("|", "\\|")
            report_lines.append(f"| {txt_a} | {txt_b} | {note} |")
        report_lines.append("")
    else:
        report_lines.append("*Cross-source contradiction checks revealed no direct factual contradictions in verified claims.*\n")

    # 4. Evidence Gaps & Thin Evidence
    report_lines.append("## 4. Evidence Gaps & Limitations\n")
    skipped_subqs = [sq for sq in subquestions if sq.status == "skipped_budget"]
    if skipped_subqs:
        for sq in skipped_subqs:
            report_lines.append(f"- **Budget Skipped:** Sub-question *'{sq.text}'* was skipped to preserve token budget.")
    else:
        report_lines.append("- All planned sub-questions were researched within budget.")
    report_lines.append("\n")

    # 5. Dropped Claims (Audit Trail)
    report_lines.append("## 5. Dropped & Unverified Claims (Audit Trail)\n")
    if dropped_claims:
        report_lines.append("| Dropped Claim | Audit Rejection Reason |")
        report_lines.append("|---|---|")
        for dc in dropped_claims:
            clean_claim = dc.text.replace("\n", " ").replace("|", "\\|")
            clean_reason = (dc.entailment_reason or "Failed verification check").replace("\n", " ").replace("|", "\\|")
            report_lines.append(f"| {clean_claim} | {clean_reason} |")
    else:
        report_lines.append("*No claims were dropped.*")
    report_lines.append("\n")

    # 6. References
    report_lines.append("## 6. References & Verified Quotes\n")
    if citations_list:
        for idx, src, quote in citations_list:
            url = src.url if src else "Unknown"
            domain = src.domain if src else "unknown"
            fetched = src.fetched_at.strftime("%Y-%m-%d %H:%M UTC") if src and src.fetched_at else "N/A"
            clean_quote = quote.replace("\n", " ")
            report_lines.append(f"**[{idx}] {domain}** — [{url}]({url}) *(Fetched: {fetched})*")
            report_lines.append(f"> \"{clean_quote}\"\n")
    else:
        report_lines.append("*No citations available.*")
    report_lines.append("\n")

    # 7. Run Metadata

    latency_sec = usage["latency_ms"] / 1000.0
    accounted = usage["budget_accounted_tokens"]

    report_lines.append("## 7. Run Metadata & Cost Ledger\n")
    report_lines.append(
        f"- **Provider-Reported Tokens:** {usage['provider_reported_tokens']:,} "
        f"({usage['provider_reported_tokens_in']:,} input, {usage['provider_reported_tokens_out']:,} output)"
    )
    if usage["legacy_recorded_tokens"]:
        report_lines.append(
            f"- **Legacy Recorded Tokens:** {usage['legacy_recorded_tokens']:,} (written before usage provenance was tracked; "
            "may include local estimates)"
        )
    report_lines.append(
        f"- **Attempts With Unknown Usage:** {usage['unknown_usage_attempts']} "
        f"(provider reported no usage; {usage['unknown_usage_reserved_tokens']:,} tokens reserved and charged to the budget, not known to be spent)"
    )
    report_lines.append(f"- **Estimated Input Tokens (local estimate, not provider usage):** {usage['estimated_input_tokens']:,}")
    report_lines.append(
        f"- **Token Budget Accounted:** {accounted:,} / {job.max_tokens:,} ({(accounted / max(1, job.max_tokens)) * 100:.1f}%) "
        "- provider-reported + legacy + retained reservations"
    )
    report_lines.append(
        f"- **Provider Attempts:** {usage['provider_attempts']} ({usage['failed_attempts']} failed, "
        f"{usage['in_flight_or_interrupted_attempts']} interrupted); cache hits: {usage['cache_hits']} (excluded from spend)"
    )
    report_lines.append(f"- **Modeled USD Spend (list price, provider-reported tokens only):** ${usage['modeled_usd_reported']:.4f}")
    if usage["modeled_usd_legacy"]:
        report_lines.append(f"- **Modeled USD, Legacy Rows (unverified):** ${usage['modeled_usd_legacy']:.4f}")
    report_lines.append(f"- **Total Elapsed Time (summed provider latency):** {latency_sec:.2f} seconds")
    report_lines.append("- **Models Used:** Writer (`llama-3.3-70b-versatile`), Verifier (`gemma2-9b-it`)")

    markdown = "\n".join(report_lines)
    job.report_md = markdown
    await db.commit()
    return markdown

