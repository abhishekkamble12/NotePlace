# app/fetcher.py
"""
Backwards-compatibility re-export module.
All core fetching and SSRF protection functionality lives in app.core.fetcher.
"""
from app.core.fetcher import (
    MAX_PAGE_BYTES,
    REQUEST_TIMEOUT,
    SSRFSecurityError,
    chunk_text,
    fetch_url_snapshot,
    is_ip_allowed,
    store_evidence_source,
    validate_url_security,
)

__all__ = [
    "MAX_PAGE_BYTES",
    "REQUEST_TIMEOUT",
    "SSRFSecurityError",
    "chunk_text",
    "fetch_url_snapshot",
    "is_ip_allowed",
    "store_evidence_source",
    "validate_url_security",
]
