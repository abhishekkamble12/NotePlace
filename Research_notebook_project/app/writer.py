# app/writer.py
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import call_llm
from app.models import Claim

class DraftClaim(BaseModel):
    claim: str = Field(
        description="An atomic factual claim stating exactly one verifiable fact."
    )
    source_id: int = Field(
        description="The source_id of the source text containing the exact quote."
    )
    exact_quote: str = Field(
        description="The verbatim, character-for-character quote copied directly from the evidence text."
    )

class SubQuestionDraft(BaseModel):
    claims: list[DraftClaim] = Field(
        min_length=1,
        max_length=4,
        description="2 to 4 atomic claims supported directly by verbatim quotes from the evidence."
    )

WRITER_SYSTEM_PROMPT = """You are a rigorous, fact-based scientific research writer.
Your job is to draft atomic claims to answer a specific sub-question, supported entirely by verbatim quotes from the provided evidence.

CRITICAL RULES:
1. Every claim MUST be atomic: state exactly one verifiable fact.
2. "exact_quote" MUST be copied verbatim, character-for-character from the evidence snippet. Do NOT paraphrase, summarize, or alter even a single word in the quote.
3. "source_id" MUST match the exact source_id associated with that quote.
4. Do not speculate or introduce outside knowledge. If the provided evidence does not answer the question, state only what is provable from the text.
5. Never bundle multiple assertions into one claim."""

async def draft_claims_for_subquestion(
    db: AsyncSession,
    job_id: int,
    subq_id: int,
    subq_text: str,
    evidence_snippets: list[dict], # [{"source_id": int, "text": str}]
    writer_model: str = "llama-3.3-70b-versatile"
) -> list[Claim]:
    """
    Drafts structured (claim, source_id, exact_quote) tuples for a sub-question
    under the 'writing' budget phase.
    """
    evidence_formatted = []
    for i, snip in enumerate(evidence_snippets, 1):
        evidence_formatted.append(
            f"--- EVIDENCE CHUNK {i} (source_id: {snip['source_id']}) ---\n{snip['text']}\n"
        )
    evidence_block = "\n".join(evidence_formatted)

    user_prompt = f"""SUB-QUESTION:
\"{subq_text}\"

EVIDENCE CHUNKS:
{evidence_block}

Extract 2 to 4 atomic claims answering this sub-question. Ensure each claim cites the exact source_id and copies an exact verbatim quote."""

    messages = [
        {"role": "system", "content": WRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    _, draft = await call_llm(
        db=db,
        job_id=job_id,
        phase="writing",
        model=writer_model,
        messages=messages,
        max_tokens=800,
        schema=SubQuestionDraft
    )

    created_claims: list[Claim] = []
    if draft and draft.claims:
        for item in draft.claims:
            claim_row = Claim(
                job_id=job_id,
                subq_id=subq_id,
                text=item.claim,
                source_id=item.source_id,
                quote=item.exact_quote,
                final_status="pending"
            )
            db.add(claim_row)
            created_claims.append(claim_row)

        await db.commit()
        for claim_row in created_claims:
            await db.refresh(claim_row)

    return created_claims

