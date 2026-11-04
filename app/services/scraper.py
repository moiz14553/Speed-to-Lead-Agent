import asyncio
import ipaddress
import logging
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.config import get_settings

logger = logging.getLogger(__name__)

FIRECRAWL_URL = "https://api.firecrawl.dev/v2/scrape"
# Share of a page's timeout Firecrawl may use, leaving time for the httpx fallback.
FIRECRAWL_TIMEOUT_SHARE = 0.6
MAX_HTML_BYTES = 2_000_000
MAX_REDIRECTS = 5
USER_AGENT = "Mozilla/5.0 (compatible; SpeedToLeadBot/1.0)"


class ScrapeError(Exception):
    pass


@dataclass
class Page:
    url: str
    html: str
    text: str
    via_firecrawl: bool = False


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    description = soup.find("meta", attrs={"name": "description"})
    parts = [
        soup.title.get_text(strip=True) if soup.title else "",
        (description.get("content") or "") if description else "",
    ]
    for tag in soup(["title", "script", "style", "noscript", "svg"]):
        tag.decompose()
    parts.append(soup.get_text(separator=" ", strip=True))
    return "\n".join(part for part in parts if part)


async def assert_public_host(host: str) -> None:
    """Domains come from user input, so never fetch private or internal addresses."""
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, 443)
    except OSError as exc:
        raise ScrapeError(f"could not resolve {host}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        ip = getattr(ip, "ipv4_mapped", None) or ip
        if not ip.is_global:
            raise ScrapeError(f"{host} resolves to a non-public address")


async def _fetch_direct(url: str) -> Page | None:
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}) as client:
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                raise ScrapeError(f"unsupported URL {url}")
            await assert_public_host(parts.hostname)

            async with client.stream("GET", url) as response:
                if response.is_redirect:
                    url = urljoin(url, response.headers["location"])
                    continue
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                if "html" not in response.headers.get("content-type", "html"):
                    raise ScrapeError(f"{url} is not an HTML page")

                body = b""
                async for chunk in response.aiter_bytes():
                    body += chunk
                    if len(body) >= MAX_HTML_BYTES:
                        break
                html = body.decode(response.encoding or "utf-8", errors="replace")
                return Page(url=url, html=html, text=html_to_text(html))
    raise ScrapeError(f"too many redirects from {url}")


async def _fetch_firecrawl(url: str, api_key: str, timeout: float) -> Page | None:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            FIRECRAWL_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "url": url,
                "formats": ["markdown", "rawHtml"],
                "timeout": max(1000, int(timeout * 1000)),
            },
        )
        response.raise_for_status()
    data = response.json().get("data") or {}
    if (data.get("metadata") or {}).get("statusCode") == 404:
        return None
    html = data.get("rawHtml") or ""
    text = data.get("markdown") or html_to_text(html)
    if not text:
        raise ScrapeError(f"Firecrawl returned no content for {url}")
    return Page(url=url, html=html, text=text, via_firecrawl=True)


async def _fetch_page(domain: str, path: str) -> Page | None:
    """Return the page, or None if it does not exist (404)."""
    settings = get_settings()
    timeout = settings.scrape_timeout_seconds
    url = f"https://{domain}{path}"

    async with asyncio.timeout(timeout):
        if settings.firecrawl_api_key is not None:
            share = timeout * FIRECRAWL_TIMEOUT_SHARE
            try:
                async with asyncio.timeout(share):
                    return await _fetch_firecrawl(
                        url, settings.firecrawl_api_key.get_secret_value(), share
                    )
            except (httpx.HTTPError, TimeoutError, ScrapeError, ValueError) as exc:
                logger.warning("Firecrawl failed for %s, using httpx: %r", url, exc)

        try:
            return await _fetch_direct(url)
        except httpx.TransportError:
            # Some sites only answer on the www host.
            return await _fetch_direct(f"https://www.{domain}{path}")


async def scrape_site(domain: str) -> tuple[list[Page], list[str]]:
    """Fetch the homepage and /about concurrently. Returns the pages and any errors."""
    results = await asyncio.gather(
        _fetch_page(domain, "/"), _fetch_page(domain, "/about"), return_exceptions=True
    )

    pages: list[Page] = []
    errors: list[str] = []
    for path, result in zip(("/", "/about"), results):
        if isinstance(result, Page):
            pages.append(result)
        elif isinstance(result, TimeoutError):
            timeout = get_settings().scrape_timeout_seconds
            errors.append(f"scrape {path}: timed out after {timeout:g}s")
        elif isinstance(result, httpx.HTTPStatusError):
            errors.append(f"scrape {path}: HTTP {result.response.status_code}")
        elif isinstance(result, (httpx.HTTPError, ScrapeError)):
            errors.append(f"scrape {path}: {str(result) or type(result).__name__}")
        elif isinstance(result, BaseException):
            raise result
        elif path == "/":
            errors.append("scrape /: homepage returned 404")
    return pages, errors
