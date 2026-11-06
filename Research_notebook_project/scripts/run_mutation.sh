#!/bin/bash
# Runs inside python:3.12-slim (see Makefile test-mutation). Mutates app/core/verifier.py only; tests that need Postgres/network are excluded.
set -e
pip install -q "mutmut<4" pytest pytest-asyncio hypothesis sqlalchemy[asyncio] asyncpg pgvector groq pydantic rapidfuzz httpx trafilatura pytest-randomly alembic langgraph mcp 2>&1 | tail -2 || true
export CITEGUARD_EVIDENCE_HMAC_KEY=test-key-not-a-secret
mutmut run || true
mutmut results
