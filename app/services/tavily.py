import asyncio

import httpx

from app.config import get_settings
from app.models.enrichment import SourceDoc

SEARCH_URL = "https://api.tavily.com/search"


class TavilyError(Exception):
    pass


async def search(query: str, max_results: int = 5) -> list[SourceDoc]:
    settings = get_settings()
    if settings.tavily_api_key is None:
        raise TavilyError("TAVILY_API_KEY is not set")

    timeout = settings.tavily_timeout_seconds
    async with asyncio.timeout(timeout), httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key.get_secret_value()}"},
            json={"query": query, "search_depth": "basic", "max_results": max_results},
        )
        response.raise_for_status()

    return [
        SourceDoc(url=item["url"], title=item.get("title"), text=item.get("content") or "")
        for item in response.json().get("results", [])
        if item.get("url")
    ]
