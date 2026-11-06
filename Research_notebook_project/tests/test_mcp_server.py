# tests/test_mcp_server.py
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.verifier import VerificationResult
from app.mcp.server import mcp_server, verify_quote


@pytest.mark.asyncio
async def test_mcp_tool_registration():
    """Verifies that the MCP server registers both research and verify_quote tools."""
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]
    assert "research" in tool_names
    assert "verify_quote" in tool_names


@pytest.mark.asyncio
async def test_mcp_verify_quote_tool():
    """Verifies direct quote verification via MCP tool."""
    mock_res = VerificationResult(
        quote_found=True,
        match_score=100.0,
        verdict="supports",
        reason="Direct verbatim match in context",
        action="kept"
    )

    # Explicit async session double: the handler awaits commit/refresh and calls add() synchronously.
    fake_db = MagicMock()
    fake_db.add = MagicMock()
    fake_db.commit = AsyncMock()
    fake_db.refresh = AsyncMock()
    session_factory = MagicMock()
    session_factory.return_value.__aenter__ = AsyncMock(return_value=fake_db)
    session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

    with patch("app.mcp.server.init_db", AsyncMock()), \
         patch("app.mcp.server.AsyncSessionLocal", session_factory), \
         patch("app.mcp.server.verify_claim", AsyncMock(return_value=mock_res)):
        out_raw = await verify_quote(
            claim="Google Sycamore sampled circuits in 200 seconds.",
            quote="Sycamore sampled circuits in 200 seconds.",
            context="Full snapshot context..."
        )
        parsed = json.loads(out_raw)
        assert parsed["quote_found"] is True
        assert parsed["entailment_verdict"] == "supports"
        assert parsed["action"] == "kept"
        assert parsed["source_authenticated"] is False
        assert parsed["span_source"] == "caller_supplied_context"
