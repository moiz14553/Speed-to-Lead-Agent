import asyncio
from email.utils import parsedate_to_datetime

import httpx

from app.config import get_settings
from app.models.enrichment import SourceDoc

SEARCH_URL = "https://api.tavily.com/search"


class TavilyError(Exception):
    pass


def _iso_date(text: str | None) -> str | None:
    """Convert Tavily's 'Tue, 11 Mar 2025 17:00:00 GMT' to YYYY-MM-DD."""
    try:
        return parsedate_to_datetime(text).date().isoformat() if text else None
    except (TypeError, ValueError):
        return None


async def search(query: str, max_results: int = 5) -> list[SourceDoc]:
    return await _post({"query": query, "search_depth": "basic", "max_results": max_results})


async def search_news(query: str, max_results: int = 5) -> list[SourceDoc]:
    """Search news from the last year."""
    return await _post(
        {
            "query": query,
            "search_depth": "basic",
            "max_results": max_results,
            "topic": "news",
            "time_range": "year",
        }
    )


async def _post(payload: dict) -> list[SourceDoc]:
    settings = get_settings()
    if settings.tavily_api_key is None:
        raise TavilyError("TAVILY_API_KEY is not set")

    timeout = settings.tavily_timeout_seconds
    async with asyncio.timeout(timeout), httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            SEARCH_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key.get_secret_value()}"},
            json=payload,
        )
        response.raise_for_status()

    return [
        SourceDoc(
            url=item["url"],
            title=item.get("title"),
            published_date=_iso_date(item.get("published_date")),
            text=item.get("content") or "",
        )
        for item in response.json().get("results", [])
        if item.get("url")
    ]
