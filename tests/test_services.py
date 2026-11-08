import asyncio
import csv
import importlib.util
import socket
from pathlib import Path

import httpx
import pytest
from pydantic import SecretStr

from app.services import scraper, tavily
from app.services.tech_detect import detect_tech

_spec = importlib.util.spec_from_file_location(
    "test_enrich_script", Path(__file__).parent.parent / "scripts" / "test_enrich.py"
)
enrich_script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enrich_script)


@pytest.mark.parametrize(
    "html,expected",
    [
        ('<script src="https://cdn.shopify.com/s/x.js"></script>', "Shopify"),
        ('<link href="/wp-content/themes/a/style.css" rel="stylesheet">', "WordPress"),
        ('<meta name="generator" content="WordPress 6.5">', "WordPress"),
        ('<script src="/wp-content/plugins/woocommerce/a.js"></script>', "WooCommerce"),
        ('<script src="//js.hs-scripts.com/123.js"></script>', "HubSpot"),
        (
            '<script src="https://www.googletagmanager.com/gtag/js?id=G-1"></script>',
            "Google Analytics",
        ),
        ("<script>(function(){a.src='https://connect.facebook.net/en_US/fbevents.js'})()</script>",
         "Meta Pixel"),
    ],
)
def test_detects_signature(html, expected):
    assert expected in detect_tech(html)


def test_facebook_sdk_alone_is_not_meta_pixel():
    html = '<script src="https://connect.facebook.net/en_US/sdk.js"></script>'
    assert "Meta Pixel" not in detect_tech(html)


def test_page_text_mentioning_a_product_is_not_a_match():
    html = "<html><body><p>We migrate stores from cdn.shopify.com and /wp-content/.</p></body></html>"
    assert detect_tech(html) == []


def test_detects_several_on_one_page():
    html = (
        '<script src="https://cdn.shopify.com/a.js"></script>'
        '<script src="https://static.klaviyo.com/b.js"></script>'
    )
    assert detect_tech(html) == ["Shopify", "Klaviyo"]


def test_html_to_text_keeps_title_and_description_and_drops_scripts():
    html = (
        '<html><head><title>Acme</title><meta name="description" content="Widgets"></head>'
        "<body><script>var secret = 1;</script><p>Hello</p></body></html>"
    )
    text = scraper.html_to_text(html)
    assert "Acme" in text and "Widgets" in text and "Hello" in text
    assert "secret" not in text


def _fake_addrinfo(address: str):
    async def getaddrinfo(self, host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))]

    return getaddrinfo


@pytest.mark.parametrize(
    "address", ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "::1", "::ffff:10.0.0.1"]
)
async def test_private_addresses_are_blocked(monkeypatch, address):
    monkeypatch.setattr(type(asyncio.get_running_loop()), "getaddrinfo", _fake_addrinfo(address))

    with pytest.raises(scraper.ScrapeError, match="non-public"):
        await scraper.assert_public_host("internal.example.com")


async def test_public_address_is_allowed(monkeypatch):
    monkeypatch.setattr(type(asyncio.get_running_loop()), "getaddrinfo", _fake_addrinfo("8.8.8.8"))

    await scraper.assert_public_host("example.com")


REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_transport(monkeypatch, handler):
    def client(**kwargs):
        return REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)


@pytest.fixture
def public_hosts(monkeypatch):
    checked = []

    async def check(host):
        checked.append(host)
        if host == "internal.example.com":
            raise scraper.ScrapeError(f"{host} resolves to a non-public address")

    monkeypatch.setattr(scraper, "assert_public_host", check)
    return checked


async def test_scrape_fetches_homepage_and_about(monkeypatch, public_hosts):
    def handler(request):
        return httpx.Response(200, html=f"<title>{request.url.path}</title>")

    _mock_transport(monkeypatch, handler)

    pages, errors = await scraper.scrape_site("acme.com")

    assert sorted(page.url for page in pages) == ["https://acme.com/", "https://acme.com/about"]
    assert pages[0].url == "https://acme.com/"
    assert errors == []


async def test_missing_about_page_is_not_an_error(monkeypatch, public_hosts):
    def handler(request):
        if request.url.path == "/about":
            return httpx.Response(404)
        return httpx.Response(200, html="<title>Home</title>")

    _mock_transport(monkeypatch, handler)

    pages, errors = await scraper.scrape_site("acme.com")

    assert [page.url for page in pages] == ["https://acme.com/"]
    assert errors == []


async def test_blocked_homepage_is_reported(monkeypatch, public_hosts):
    _mock_transport(monkeypatch, lambda request: httpx.Response(403))

    pages, errors = await scraper.scrape_site("acme.com")

    assert pages == []
    assert len(errors) == 2
    assert errors[0] == "scrape /: HTTP 403"


async def test_redirect_to_private_host_is_blocked(monkeypatch, public_hosts):
    def handler(request):
        return httpx.Response(302, headers={"location": "http://internal.example.com/admin"})

    _mock_transport(monkeypatch, handler)

    pages, errors = await scraper.scrape_site("acme.com")

    assert pages == []
    assert "non-public" in errors[0]
    assert "internal.example.com" in public_hosts


async def test_redirect_uses_final_url(monkeypatch, public_hosts):
    def handler(request):
        if request.url.host == "acme.com":
            return httpx.Response(301, headers={"location": f"https://www.acme.com{request.url.path}"})
        return httpx.Response(200, html="<title>Home</title>")

    _mock_transport(monkeypatch, handler)

    pages, _ = await scraper.scrape_site("acme.com")

    assert pages[0].url == "https://www.acme.com/"


async def test_scrape_timeout_is_reported(monkeypatch, public_hosts, settings):
    settings.scrape_timeout_seconds = 0.01

    async def handler(request):
        await asyncio.sleep(5)

    _mock_transport(monkeypatch, handler)

    pages, errors = await scraper.scrape_site("acme.com")

    assert pages == []
    assert "scrape /: timed out after 0.01s" in errors


async def test_non_html_is_rejected(monkeypatch, public_hosts):
    _mock_transport(
        monkeypatch,
        lambda request: httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}),
    )

    pages, errors = await scraper.scrape_site("acme.com")

    assert pages == []
    assert "not an HTML page" in errors[0]


async def test_firecrawl_used_when_key_set(monkeypatch, public_hosts, settings):
    settings.firecrawl_api_key = SecretStr("fc-key")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "success": True,
                "data": {
                    "markdown": "# Acme",
                    "rawHtml": '<script src="https://cdn.shopify.com/a.js"></script>',
                    "metadata": {"statusCode": 200},
                },
            },
        )

    _mock_transport(monkeypatch, handler)

    pages, _ = await scraper.scrape_site("acme.com")

    assert all(page.via_firecrawl for page in pages)
    assert pages[0].text == "# Acme"
    assert requests[0].url.host == "api.firecrawl.dev"
    assert requests[0].headers["authorization"] == "Bearer fc-key"


async def test_firecrawl_failure_falls_back_to_httpx(monkeypatch, public_hosts, settings):
    settings.firecrawl_api_key = SecretStr("fc-key")

    def handler(request):
        if request.url.host == "api.firecrawl.dev":
            return httpx.Response(500)
        return httpx.Response(200, html="<title>Direct</title>")

    _mock_transport(monkeypatch, handler)

    pages, errors = await scraper.scrape_site("acme.com")

    assert not any(page.via_firecrawl for page in pages)
    assert pages[0].text == "Direct"
    assert errors == []


async def test_tavily_search_parses_results(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={"results": [{"url": "https://a.com", "title": "A", "content": "About A"}]},
        )

    _mock_transport(monkeypatch, handler)

    docs = await tavily.search("acme")

    assert docs[0].url == "https://a.com"
    assert docs[0].text == "About A"
    assert requests[0].headers["authorization"] == "Bearer test-tavily"


async def test_tavily_without_key_raises(settings):
    settings.tavily_api_key = None

    with pytest.raises(tavily.TavilyError, match="TAVILY_API_KEY"):
        await tavily.search("acme")


async def test_tavily_timeout(monkeypatch, settings):
    settings.tavily_timeout_seconds = 0.01

    async def handler(request):
        await asyncio.sleep(5)

    _mock_transport(monkeypatch, handler)

    with pytest.raises(TimeoutError):
        await tavily.search("acme")


def _lead_json(provider="gemini", model="gemini-3.8-flash") -> dict:
    fact = {"value": "x", "source_url": "https://a.com", "confidence": "high"}
    return {
        **{field: fact for field in enrich_script.SINGLE_FIELDS},
        "tech_stack": [fact, fact],
        "recent_news": [],
        "is_free_email": False,
        "cost_usd": 0.01,
        "latency_ms": 5,
        "llm_provider": provider,
        "llm_model": model,
        "errors": ["one", "two"],
    }


def test_results_row_has_news_confidence_and_dates():
    lead = _lead_json()
    lead["recent_news"] = [
        {"value": "a", "source_url": "https://a.com", "confidence": "medium", "published_date": "2026-10-01"},
        {"value": "b", "source_url": "https://b.com", "confidence": "low", "published_date": "2026-09"},
    ]

    row = enrich_script.to_columns(lead)

    assert row["recent_news_confidences"] == "medium | low"
    assert row["recent_news_dates"] == "2026-10-01 | 2026-09"
    assert set(row) <= set(enrich_script.COLUMNS)


def test_results_row_has_provider_model_and_sources():
    row = enrich_script.to_columns(_lead_json("anthropic", "claude-haiku-5-5"))

    assert row["llm_provider"] == "anthropic"
    assert row["llm_model"] == "claude-haiku-5-5"
    assert row["industry_source"] == "https://a.com"
    assert row["tech_stack"] == "x | x"
    assert row["errors"] == "one | two"
    assert set(row) <= set(enrich_script.COLUMNS)


REAL_HTTP_CLIENT = httpx.Client


def _run_script(monkeypatch, input_file, output_file, handler):
    monkeypatch.setattr(
        enrich_script.httpx,
        "Client",
        lambda **kwargs: REAL_HTTP_CLIENT(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setattr(
        "sys.argv", ["test_enrich.py", "--input", str(input_file), "--output", str(output_file)]
    )
    enrich_script.main()


def test_runs_append_so_they_can_be_compared(monkeypatch, tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("email,name,company_hint\njo@acme.com,Jo,\n", encoding="utf-8")
    output_file = tmp_path / "out.csv"

    _run_script(
        monkeypatch, input_file, output_file, lambda r: httpx.Response(200, json=_lead_json())
    )
    _run_script(
        monkeypatch,
        input_file,
        output_file,
        lambda r: httpx.Response(200, json=_lead_json("anthropic", "claude-haiku-5-5")),
    )

    with output_file.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [row["llm_provider"] for row in rows] == ["gemini", "anthropic"]
    assert [row["llm_model"] for row in rows] == ["gemini-3.8-flash", "claude-haiku-5-5"]


def test_failed_row_is_kept_with_provider(monkeypatch, tmp_path):
    input_file = tmp_path / "in.csv"
    input_file.write_text("email,name,company_hint\njo@acme.com,Jo,\n", encoding="utf-8")
    output_file = tmp_path / "out.csv"
    detail = {"message": "LLM failed: x", "llm_provider": "gemini", "llm_model": "m"}

    _run_script(
        monkeypatch, input_file, output_file, lambda r: httpx.Response(502, json={"detail": detail})
    )

    with output_file.open(newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    assert row["http_status"] == "502"
    assert row["llm_provider"] == "gemini"
    assert row["errors"] == "LLM failed: x"
