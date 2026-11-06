# app/verifier.py
"""
Backwards-compatibility re-export module.
All core verification and rewrite functionality lives in app.core.verifier.
"""
from app.core.verifier import (
    REWRITER_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    EntailmentVerdict,
    RewrittenClaim,
    VerificationResult,
    check_quote_in_snapshot,
    normalize_text,
    rewrite_claim,
    verify_claim,
)

__all__ = [
    "REWRITER_SYSTEM_PROMPT",
    "VERIFIER_SYSTEM_PROMPT",
    "EntailmentVerdict",
    "RewrittenClaim",
    "VerificationResult",
    "check_quote_in_snapshot",
    "normalize_text",
    "rewrite_claim",
    "verify_claim",
]
