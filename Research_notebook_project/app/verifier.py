# app/verifier.py
import re
import unicodedata
from typing import Literal
from pydantic import BaseModel, Field
from rapidfuzz import fuzz
from sqlalchemy.ext.asyncio import AsyncSession
from app.llm import call_llm

class EntailmentVerdict(BaseModel):
    verdict: Literal["supports", "partial", "unsupported"] = Field(
        description="Whether the quote fully supports, partially supports, or does not support the claim"
    )
    reason: str = Field(
        description="One concise line explaining the verdict, especially identifying added numbers, overgeneralizations, or unsupported causal leaps."
    )

class VerificationResult(BaseModel):
    quote_found: bool
    match_score: float
    verdict: Literal["supports", "partial", "unsupported"]
    reason: str
    action: Literal["kept", "needs_rewrite", "dropped"]


def normalize_text(text: str) -> str:
    """
    Normalizes text for deterministic quote matching:
    1. Unicode NFKC normalization
    2. Lowercase
    3. Standardize quotes, apostrophes, and dashes
    4. Collapse whitespace
    """
    if not text:
        return ""
    # Unicode NFKC
    norm = unicodedata.normalize("NFKC", text).lower()
    # Replace curly quotes and variations
    norm = norm.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("`", "'")
    # Replace various dash types with standard hyphen
    norm = norm.replace("—", "-").replace("–", "-")
    # Collapse all whitespace and newlines into single spaces
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def check_quote_in_snapshot(quote: str, snapshot: str) -> tuple[bool, float]:
    """
    Check 1: Deterministic search.
    First tries strict substring search on normalized text.
    Falls back to rapidfuzz.partial_ratio (threshold >= 95.0) for minor formatting differences.
    Returns (found: bool, match_score: float).
    """
    norm_quote = normalize_text(quote)
    norm_snapshot = normalize_text(snapshot)

    if not norm_quote or not norm_snapshot:
        return False, 0.0

    # 1. Exact normalized substring match
    if norm_quote in norm_snapshot:
        return True, 100.0

    # 2. Fuzzy fallback (partial ratio)
    score = fuzz.partial_ratio(norm_quote, norm_snapshot)
    if score >= 95.0:
        return True, float(score)

    return False, float(score)


def extract_surrounding_context(quote: str, snapshot: str, window_chars: int = 400) -> str:
    """Finds the quote in the snapshot and extracts surrounding context (~1 paragraph)."""
    norm_quote = normalize_text(quote)
    norm_snapshot = normalize_text(snapshot)

    idx = norm_snapshot.find(norm_quote)
    if idx != -1:
        start = max(0, idx - window_chars)
        end = min(len(snapshot), idx + len(quote) + window_chars)
        return snapshot[start:end].strip()

    # Fallback to first 800 chars of snapshot if exact position unclear
    return snapshot[:800].strip()


VERIFIER_SYSTEM_PROMPT = """You are a rigorous, impartial citation verification auditor.
Your sole job is to verify whether a specific CLAIM is strictly supported by an EXACT QUOTE taken from a source document.

You will be given:
1. The Claim
2. The Exact Quote
3. Surrounding Context from the source

Carefully evaluate the claim against the quote and context:
- "supports": The quote directly and completely supports every part of the claim.
- "partial": The quote supports the main thrust, but the claim adds minor unproven details or weakens the qualifier.
- "unsupported": The claim makes assertions not in the quote, alters numbers/dates/entities, turns correlation into causation, or overgeneralizes (e.g. from "some" to "all/always").

Be strict. LLMs frequently hallucinate facts into citations."""


async def verify_claim(
    db: AsyncSession,
    job_id: int,
    claim_text: str,
    quote: str,
    snapshot: str,
    verifier_model: str = "gemma2-9b-it"
) -> VerificationResult:
    """
    Two-stage verification engine:
      Check 1: Deterministic quote search in snapshot (0 LLM cost).
               If quote is missing, immediately drops with zero tokens.
      Check 2: Cross-model entailment check (claim + quote + context).
    """
    # ----------------------------------------------------
    # Check 1: Deterministic Check
    # ----------------------------------------------------
    found, score = check_quote_in_snapshot(quote, snapshot)
    if not found:
        return VerificationResult(
            quote_found=False,
            match_score=score,
            verdict="unsupported",
            reason=f"Quote not found in source snapshot (fuzzy match score: {score:.1f}% < 95.0%)",
            action="dropped"
        )

    # ----------------------------------------------------
    # Check 2: Cross-Model Entailment Check
    # ----------------------------------------------------
    context = extract_surrounding_context(quote, snapshot)
    user_prompt = f"""CLAIM:
\"\"\"{claim_text}\"\"\"

EXACT QUOTE:
\"\"\"{quote}\"\"\"

SURROUNDING CONTEXT:
\"\"\"{context}\"\"\"

Verify whether the claim is entailed by the quote. Return your verdict and reason."""

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    # Call LLM via centralized gateway under 'verifying' phase
    _, parsed = await call_llm(
        db=db,
        job_id=job_id,
        phase="verifying",
        model=verifier_model,
        messages=messages,
        max_tokens=250,
        schema=EntailmentVerdict
    )

    if not parsed:
        return VerificationResult(
            quote_found=True,
            match_score=score,
            verdict="unsupported",
            reason="Verifier model failed to produce a structured verdict",
            action="dropped"
        )

    action: Literal["kept", "needs_rewrite", "dropped"]
    if parsed.verdict == "supports":
        action = "kept"
    elif parsed.verdict == "partial":
        action = "needs_rewrite"
    else:
        action = "dropped"

    return VerificationResult(
        quote_found=True,
        match_score=score,
        verdict=parsed.verdict,
        reason=parsed.reason,
        action=action
    )


class RewrittenClaim(BaseModel):
    rewritten_claim: str = Field(
        description="The revised atomic claim that strictly asserts ONLY what is directly supported by the quote, removing unproven qualifiers, dates, or causal leaps."
    )

REWRITER_SYSTEM_PROMPT = """You are a precision rewriter.
Your task is to rewrite a claim so that it is 100% truthful to an exact quote, addressing the verifier's criticism.
Rules:
1. Do not add ANY details not present in the quote.
2. Tone down or remove overgeneralizations and unsupported claims.
3. Make the claim directly provable by the quote alone."""

async def rewrite_claim(
    db: AsyncSession,
    job_id: int,
    original_claim: str,
    quote: str,
    critique_reason: str,
    writer_model: str = "llama-3.3-70b-versatile"
) -> str | None:
    """Attempts a single rewrite of a partial or contested claim."""
    user_prompt = f"""ORIGINAL CLAIM:
\"{original_claim}\"

EXACT QUOTE:
\"{quote}\"

VERIFIER CRITIQUE:
\"{critique_reason}\"

Rewrite the claim so that it is strictly and completely supported by the quote."""

    messages = [
        {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]

    _, parsed = await call_llm(
        db=db,
        job_id=job_id,
        phase="verifying",
        model=writer_model,
        messages=messages,
        max_tokens=300,
        schema=RewrittenClaim
    )
    return parsed.rewritten_claim if parsed else None


async def run_verification_loop(
    db: AsyncSession,
    job_id: int,
    shadow: bool = False
) -> None:
    """
    Executes the verification loop for all claims in a job:
    1. Fetches all claims with their associated source snapshots.
    2. Runs Check 1 (Deterministic) and Check 2 (Cross-Model Entailment).
    3. If 'needs_rewrite', attempts 1 rewrite and re-verifies.
    4. If shadow=True, records shadow_verdict and keeps all claims.
    """
    from sqlalchemy import select
    from app.models import Claim, Source

    # Fetch claims and associated sources
    result = await db.execute(
        select(Claim, Source.snapshot)
        .outerjoin(Source, Claim.source_id == Source.id)
        .where(Claim.job_id == job_id)
    )
    rows = result.all()

    for claim, snapshot in rows:
        snapshot_text = snapshot or ""
        v_result = await verify_claim(
            db=db,
            job_id=job_id,
            claim_text=claim.text,
            quote=claim.quote,
            snapshot=snapshot_text
        )

        claim.quote_found = v_result.quote_found
        claim.entailment = v_result.verdict
        claim.entailment_reason = v_result.reason

        if shadow:
            # Shadow mode: record verdict without dropping or rewriting
            claim.shadow_verdict = v_result.verdict
            claim.final_status = "kept"
            continue

        if v_result.action == "kept":
            claim.final_status = "kept"
        elif v_result.action == "needs_rewrite" and claim.rewrite_count == 0:
            # Attempt 1 rewrite
            rewritten_text = await rewrite_claim(
                db=db,
                job_id=job_id,
                original_claim=claim.text,
                quote=claim.quote,
                critique_reason=v_result.reason
            )
            if rewritten_text:
                claim.text = rewritten_text
                claim.rewrite_count += 1
                # Re-verify the rewritten claim
                re_v_result = await verify_claim(
                    db=db,
                    job_id=job_id,
                    claim_text=claim.text,
                    quote=claim.quote,
                    snapshot=snapshot_text
                )
                claim.entailment = re_v_result.verdict
                claim.entailment_reason = re_v_result.reason
                if re_v_result.verdict == "supports":
                    claim.final_status = "kept"
                else:
                    claim.final_status = "dropped"
            else:
                claim.final_status = "dropped"
        else:
            claim.final_status = "dropped"

    await db.commit()
