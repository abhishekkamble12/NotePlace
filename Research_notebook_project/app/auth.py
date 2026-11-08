"""Who is calling. One dependency, so the identity provider can change without touching endpoints.

Today: static bearer tokens from CITEGUARD_DEV_TOKENS="token1:alice,token2:bob" (token -> user subject).
Fail closed: unset or empty means every notebook/chat request is 401.
shortcut: static tokens, no expiry or rotation; replace with Cognito JWT verification (Week 4) before any public deploy.
"""
import hmac
import os

from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.ingest.service import get_or_create_user
from app.models import User


def _subject_for(token: str) -> str | None:
    for pair in os.environ.get("CITEGUARD_DEV_TOKENS", "").split(","):
        tok, _, sub = pair.strip().partition(":")
        if tok and sub and hmac.compare_digest(tok.encode(), token.encode()):
            return sub
    return None


async def get_current_user(authorization: str | None = Header(default=None), db: AsyncSession = Depends(get_db)) -> User:
    scheme, _, token = (authorization or "").partition(" ")
    sub = _subject_for(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
    if sub is None:
        raise HTTPException(status_code=401, detail="Unauthorized", headers={"WWW-Authenticate": "Bearer"})
    user = await get_or_create_user(db, sub)
    await db.commit()
    return user
