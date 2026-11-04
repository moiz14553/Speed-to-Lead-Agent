import pytest
from pydantic import SecretStr

from app.config import Settings, get_settings
from app.models.enrichment import ExtractionResult, Sourced, SourceDoc
from app.services import scraper, tavily

HOMEPAGE = "https://acme.com/"
SEARCH_URL = "https://news.example.org/acme"


def fact(value: str, url: str = HOMEPAGE) -> Sourced:
    return Sourced(value=value, source_url=url, confidence="high")


def full_extraction() -> ExtractionResult:
    return ExtractionResult(
        company_name=fact("Acme"),
        website_summary=fact("Acme makes widgets."),
        industry=fact("Manufacturing"),
        employee_range=fact("51-200"),
        location=fact("Toronto"),
        funding=fact("Series A", SEARCH_URL),
        recent_news=[fact("Acme opens a new plant", SEARCH_URL)],
    )


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    """Ignore the developer's .env and real keys. Tests change this object directly."""
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    for name in (
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_RATES",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "ANTHROPIC_API_KEY",
        "TAVILY_API_KEY",
        "FIRECRAWL_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    settings.google_api_key = SecretStr("test-google")
    settings.anthropic_api_key = SecretStr("test-anthropic")
    settings.tavily_api_key = SecretStr("test-tavily")
    settings.llm_backoff_base_seconds = 0
    yield settings
    get_settings.cache_clear()


@pytest.fixture
def external(monkeypatch):
    """Replace every external call. Tests can reassign the functions on this object."""

    class External:
        search_queries: list[str] = []
        scraped_domains: list[str] = []
        llm_calls = 0
        llm_results: list = []

    fake = External()
    fake.search_queries = []
    fake.scraped_domains = []
    fake.llm_calls = 0
    fake.llm_results = []

    async def fake_search(query: str, max_results: int = 5):
        fake.search_queries.append(query)
        return [SourceDoc(url=SEARCH_URL, title="Acme news", text="Acme raised a Series A.")]

    html = '<html><script src="https://cdn.shopify.com/s/app.js"></script>Acme</html>'

    async def fake_scrape(domain: str):
        fake.scraped_domains.append(domain)
        page = scraper.Page(url=HOMEPAGE, html=html, text="Acme makes widgets.")
        return [page], []

    async def fake_extract(target, sources):
        fake.llm_calls += 1
        result = fake.llm_results.pop(0) if fake.llm_results else full_extraction()
        if isinstance(result, Exception):
            raise result
        return result, 0.01, None

    monkeypatch.setattr(tavily, "search", fake_search)
    monkeypatch.setattr(scraper, "scrape_site", fake_scrape)
    monkeypatch.setattr("app.services.llm.extract_facts", fake_extract)
    return fake
