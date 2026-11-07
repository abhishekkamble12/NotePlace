# app/core/search.py
"""
Search provider gateway for CiteGuard.
Tavily first, then Brave. With no working provider it returns no URLs (the sub-question gets no sources);
it never invents sources.
"""
import os
from typing import Literal

import httpx

SearchProvider = Literal["tavily", "brave"]

DEFAULT_TOP_N = 3



async def search_tavily(query: str, top_n: int = DEFAULT_TOP_N) -> list[str]:
    """Queries Tavily Search API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": top_n,
                "search_depth": "basic"
            }
        )
        if resp.status_code == 200:
            data = resp.json()
            return [r["url"] for r in data.get("results", []) if "url" in r]
    return []


async def search_brave(query: str, top_n: int = DEFAULT_TOP_N) -> list[str]:
    """Queries Brave Search API."""
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return []

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": api_key, "Accept": "application/json"},
            params={"q": query, "count": top_n}
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get("web", {}).get("results", [])
            return [r["url"] for r in results if "url" in r]
    return []


async def search_web(query: str, top_n: int = DEFAULT_TOP_N) -> list[str]:
    """
    Unified search entry point. Tries Tavily, then Brave; returns [] if neither is configured or both fail.
    """
    if os.getenv("TAVILY_API_KEY"):
        try:
            urls = await search_tavily(query, top_n=top_n)
            if urls:
                return urls[:top_n]
        except Exception:
            pass

    if os.getenv("BRAVE_API_KEY"):
        try:
            urls = await search_brave(query, top_n=top_n)
            if urls:
                return urls[:top_n]
        except Exception:
            pass

    return []
