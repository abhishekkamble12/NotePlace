# app/fetcher.py
import re
import socket
import ipaddress
import hashlib
from urllib.parse import urlparse
from datetime import datetime, timezone
import httpx
import trafilatura
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models import Source

class SSRFSecurityError(Exception):
    """Raised when an attempt is made to fetch an unsafe or private IP address."""
    pass

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
        raise SSRFSecurityError("URL missing hostname")

    # Resolve hostname to IP addresses
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise SSRFSecurityError(f"Failed to resolve hostname '{hostname}': {e}")

    for item in addr_info:
        ip_str = item[4][0]
        if not is_ip_allowed(ip_str):
            raise SSRFSecurityError(f"Security block: Host '{hostname}' resolved to disallowed/private IP '{ip_str}'")

    return hostname.lower()


async def fetch_and_clean_page(url: str) -> tuple[str, str] | None:
    """
    Fetches a webpage with SSRF guard and extracts clean article text.
    Returns (cleaned_text, domain) or None if fetch fails.
    """
    try:
        domain = validate_url_security(url)
    except SSRFSecurityError:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 CiteGuardBot/1.0"
    }

    try:
        async with httpx.AsyncClient(
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
            headers=headers
        ) as client:
            response = await client.get(url)

            # Re-validate destination URL after any redirects
            validate_url_security(str(response.url))

            if response.status_code != 200:
                return None

            # Enforce max content size limit
            content_bytes = response.content
            if len(content_bytes) > MAX_PAGE_BYTES:
                content_bytes = content_bytes[:MAX_PAGE_BYTES]

            html = content_bytes.decode(response.encoding or "utf-8", errors="ignore")

            # Extract clean readable text with trafilatura
            cleaned_text = trafilatura.extract(
                html,
                include_links=False,
                include_images=False,
                favor_precision=True,
                output_format="txt"
            )

            if not cleaned_text or len(cleaned_text.strip()) < 100:
                return None

            return cleaned_text.strip(), domain

    except Exception:
        return None


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> list[str]:
    """
    Chunks text by approximate word count with overlap.
    Averages 300-500 tokens per chunk.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk_str = " ".join(chunk_words).strip()
        if len(chunk_str) > 50:
            chunks.append(chunk_str)
        if i + chunk_size >= len(words):
            break

    return chunks


async def save_evidence_source(
    db: AsyncSession,
    job_id: int,
    url: str,
    snapshot_text: str,
    domain: str
) -> Source:
    """
    Deduplicates on (job_id, content_hash) and saves snapshot to the sources table.
    """
    content_hash = hashlib.sha256(snapshot_text.encode("utf-8")).hexdigest()

    # Check if duplicate exists for this job
    existing = await db.execute(
        select(Source).where(
            Source.job_id == job_id,
            Source.content_hash == content_hash
        )
    )
    source = existing.scalar_one_or_none()
    if source:
        return source

    source = Source(
        job_id=job_id,
        url=url,
        domain=domain,
        content_hash=content_hash,
        snapshot=snapshot_text
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source

