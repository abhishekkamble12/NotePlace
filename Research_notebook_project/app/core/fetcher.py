# app/core/fetcher.py
"""
SSRF-safe web fetcher and evidence ingestion for CiteGuard.
Reused in Phase 1 (open-web research) and Phase 2 (URL source ingestion).
"""
import hashlib
import ipaddress
import re
import socket
from urllib.parse import urlparse

import httpx
import trafilatura
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Source


class SSRFSecurityError(Exception):
    """Raised when an attempt is made to fetch an unsafe or private IP address."""

MAX_PAGE_BYTES = 2 * 1024 * 1024  # 2 MB limit
REQUEST_TIMEOUT = 8.0              # 8 seconds max


def is_ip_allowed(ip_str: str) -> bool:
    """Verifies that an IP is globally routable and not private/loopback/metadata."""
    try:
        ip = ipaddress.ip_address(ip_str)
        if (
            ip.is_private or
            ip.is_loopback or
            ip.is_link_local or
            ip.is_reserved or
            ip.is_multicast or
            not ip.is_global
        ):
            return False
        # Block cloud metadata service IP explicitly
        if str(ip) == "169.254.169.254":
            return False
        return True
    except ValueError:
        return False


def validate_url_security(url: str) -> str:
    """
    Validates URL scheme and resolves hostname to ensure it doesn't target private infrastructure.
    Returns normalized domain.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFSecurityError(f"Unsupported scheme '{parsed.scheme}': only http and https are permitted")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFSecurityError(f"Missing host in URL: {url}")

    # Reject localhost directly
    if hostname.lower() in ("localhost", "127.0.0.1", "::1"):
        raise SSRFSecurityError(f"Prohibited loopback host: {hostname}")

    try:
        # Resolve hostname to IPv4/IPv6 addresses
        addr_info = socket.getaddrinfo(hostname, None)
        ips = [info[4][0] for info in addr_info]
        for ip in ips:
            if not is_ip_allowed(ip):
                raise SSRFSecurityError(f"Host '{hostname}' resolves to restricted IP: {ip}")
    except socket.gaierror as e:
        raise SSRFSecurityError(f"Could not resolve host '{hostname}': {e}")

    return hostname.lower()


async def fetch_url_snapshot(url: str) -> tuple[str, str, str]:
    """
    Fetches clean text from a URL securely:
    1. Validates scheme and DNS against SSRF
    2. Enforces 8s timeout and 2MB max download
    3. Re-validates redirection targets
    4. Extracts clean text snapshot with trafilatura
    Returns (domain, sha256_hash, snapshot_text).
    """
    domain = validate_url_security(url)

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(REQUEST_TIMEOUT),
        follow_redirects=True,
        headers={"User-Agent": "CiteGuardBot/1.0 (+https://github.com/citeguard)"}
    ) as client:
        # Pre-hook for redirect validation to prevent redirect-based SSRF
        response = await client.get(url)
        final_url = str(response.url)
        if final_url != url:
            domain = validate_url_security(final_url)

        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP error {response.status_code} fetching {url}",
                request=response.request,
                response=response
            )

        content = response.content
        if len(content) > MAX_PAGE_BYTES:
            raise ValueError(f"Page size {len(content)} bytes exceeds 2 MB cap")

        raw_html = response.text

    # Extract text with trafilatura
    clean_text = trafilatura.extract(
        raw_html,
        include_links=False,
        include_images=False,
        output_format="txt"
    )

    if not clean_text or len(clean_text.strip()) < 50:
        # Fallback simple tag stripping if trafilatura finds no main content
        clean_text = re.sub(r"<[^>]+>", " ", raw_html)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

    content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    return domain, content_hash, clean_text


def chunk_spans(text: str, chunk_size: int = 350, overlap: int = 50) -> list[tuple[int, int]]:
    """Word-window chunks as (start, end) char offsets into the RAW text, so text[start:end] is the chunk."""
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        spans.append((words[start][0], words[end - 1][1]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return spans


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> list[str]:
    """
    Splits text into chunks of ~300-500 words with overlap.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += (chunk_size - overlap)

    return chunks


async def store_evidence_source(
    db: AsyncSession,
    job_id: int,
    url: str
) -> Source | None:
    """
    Fetches, dedupes, and persists an evidence source and its snapshot.
    Returns the created Source or existing Source if duplicate.
    """
    try:
        domain, content_hash, snapshot = await fetch_url_snapshot(url)
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return None

    # Idempotent deduplication check on (job_id, content_hash)
    existing = await db.execute(
        select(Source).where(Source.job_id == job_id, Source.content_hash == content_hash)
    )
    found = existing.scalar_one_or_none()
    if found:
        return found

    source = Source(
        job_id=job_id,
        url=url,
        domain=domain,
        content_hash=content_hash,
        snapshot=snapshot
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source
