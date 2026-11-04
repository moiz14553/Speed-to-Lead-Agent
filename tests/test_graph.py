import asyncio

import httpx
import pytest

from app.agents.enrichment import drop_unsourced, is_free_email, parse_domain, run_enrichment
from app.models.enrichment import EnrichRequest, Sourced, SourceDoc
from app.services import scraper, tavily
from app.services.llm import LLMError, LLMTimeoutError
from tests.conftest import HOMEPAGE, SEARCH_URL, fact, full_extraction


def request(email="jo@acme.com", hint=None) -> EnrichRequest:
    return EnrichRequest(email=email, name="Jo", company_hint=hint)


def without_key_fields():
    return full_extraction().model_copy(
        update={"industry": Sourced.empty(), "employee_range": Sourced.empty()}
    )


@pytest.mark.parametrize(
    "domain",
    ["gmail.com", "yahoo.com", "yahoo.co.uk", "outlook.com", "hotmail.fr", "icloud.com"],
)
def test_free_email_providers(domain):
    assert is_free_email(domain)


@pytest.mark.parametrize("domain", ["acme.com", "outlook.acme.com", "yahoo.acme.io", "gmail.co"])
def test_business_domains_are_not_free(domain):
    assert not is_free_email(domain)


@pytest.mark.parametrize(
    "hint,expected",
    [
        ("acme.com", "acme.com"),
        ("https://www.acme.com/about", "acme.com"),
        ("Acme Inc", None),
        ("Acme", None),
    ],
)
def test_parse_domain(hint, expected):
    assert parse_domain(hint) == expected


async def test_free_email_without_hint_stops_without_external_calls(external):
    lead = await run_enrichment(request("jo@gmail.com"))

    assert lead.is_free_email is True
    assert lead.domain.value is None
    assert lead.cost_usd == 0
    assert external.search_queries == []
    assert external.scraped_domains == []
    assert external.llm_calls == 0


async def test_free_email_with_domain_hint_uses_it(external):
    lead = await run_enrichment(request("jo@gmail.com", "acme.com"))

    assert lead.is_free_email is True
    assert lead.domain.value == "acme.com"
    assert lead.domain.confidence == "medium"
    assert external.scraped_domains == ["acme.com"]


async def test_free_email_with_name_hint_looks_up_website(external, monkeypatch):
    async def search(query, max_results=5):
        external.search_queries.append(query)
        return [
            SourceDoc(url="https://www.linkedin.com/company/acme", text="Acme on LinkedIn"),
            SourceDoc(url="https://www.acme.com/", text="Acme home"),
        ]

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request("jo@gmail.com", "Acme Inc"))

    assert lead.domain.value == "acme.com"
    assert lead.domain.confidence == "medium"
    assert external.scraped_domains == ["acme.com"]
    assert "official website" in external.search_queries[0]


async def test_free_email_with_free_domain_as_hint_stops(external):
    lead = await run_enrichment(request("jo@gmail.com", "gmail.com"))

    assert lead.is_free_email is True
    assert external.llm_calls == 0


async def test_business_email_ignores_hint_for_domain(external):
    lead = await run_enrichment(request("jo@acme.com", "other.com"))

    assert lead.domain.value == "acme.com"
    assert external.scraped_domains == ["acme.com"]


async def test_research_nodes_all_run(external):
    await run_enrichment(request())

    assert external.scraped_domains == ["acme.com"]
    assert len(external.search_queries) == 2
    assert any("funding" in query for query in external.search_queries)


async def test_research_runs_in_parallel(external, monkeypatch, settings):
    """Each source needs the others to have started, so this only finishes if they overlap."""
    started = asyncio.Barrier(3)
    searches = 0

    async def search(query, max_results=5):
        nonlocal searches
        searches += 1
        # The third search is the targeted retry, which runs after the research step.
        if searches <= 2:
            await started.wait()
        return []

    async def scrape(domain):
        await started.wait()
        return [], []

    monkeypatch.setattr(tavily, "search", search)
    monkeypatch.setattr(scraper, "scrape_site", scrape)
    settings.enrich_timeout_seconds = 2

    lead = await run_enrichment(request())

    assert not any(error.startswith("timeout") for error in lead.errors)


async def test_no_retry_when_key_fields_found(external):
    await run_enrichment(request())

    assert external.llm_calls == 1
    assert len(external.search_queries) == 2


async def test_retries_once_when_key_fields_missing_and_fills_them(external):
    external.llm_results = [without_key_fields(), full_extraction()]

    lead = await run_enrichment(request())

    assert external.llm_calls == 2
    assert len(external.search_queries) == 3
    assert "number of employees" in external.search_queries[2]
    assert lead.employee_range.value == "51-200"


async def test_retries_at_most_once(external):
    external.llm_results = [without_key_fields(), without_key_fields(), without_key_fields()]

    lead = await run_enrichment(request())

    assert external.llm_calls == 2
    assert lead.industry.value is None


async def test_second_extraction_does_not_lose_first_facts(external):
    second = without_key_fields().model_copy(
        update={"location": Sourced.empty(), "recent_news": []}
    )
    external.llm_results = [without_key_fields(), second]

    lead = await run_enrichment(request())

    assert lead.location.value == "Toronto"
    assert lead.recent_news[0].value == "Acme opens a new plant"


async def test_failed_search_is_reported_and_run_continues(external, monkeypatch):
    async def search(query, max_results=5):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request())

    assert "company search: boom" in lead.errors
    assert "funding search: boom" in lead.errors
    assert lead.company_name.value == "Acme"


async def test_search_http_error_is_short_and_has_no_url(external, monkeypatch):
    async def search(query, max_results=5):
        http_request = httpx.Request("POST", "https://api.tavily.com/search")
        response = httpx.Response(401, request=http_request)
        raise httpx.HTTPStatusError("long message", request=http_request, response=response)

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request())

    assert "company search: HTTP 401" in lead.errors


async def test_no_sources_skips_llm_and_returns_nulls(external, monkeypatch):
    async def search(query, max_results=5):
        raise httpx.ConnectError("boom")

    async def scrape(domain):
        return [], ["scrape /: boom"]

    monkeypatch.setattr(tavily, "search", search)
    monkeypatch.setattr(scraper, "scrape_site", scrape)

    lead = await run_enrichment(request())

    assert external.llm_calls == 0
    assert lead.company_name.value is None
    assert lead.domain.value == "acme.com"
    assert lead.domain.source_url is None
    assert lead.domain.confidence == "low"
    assert "scrape /: boom" in lead.errors


def test_drop_unsourced_removes_invented_urls():
    extraction = full_extraction().model_copy(
        update={
            "industry": fact("Made up", "https://invented.example/"),
            "recent_news": [
                fact("Real", SEARCH_URL),
                fact("Fake", "https://invented.example/news"),
            ],
        }
    )
    sources = [SourceDoc(url=HOMEPAGE, text="x"), SourceDoc(url=SEARCH_URL, text="y")]

    cleaned, dropped = drop_unsourced(extraction, sources)

    assert dropped == 2
    assert cleaned.industry == Sourced.empty()
    assert [item.value for item in cleaned.recent_news] == ["Real"]
    assert cleaned.company_name.value == "Acme"


async def test_invented_source_is_nulled_and_reported(external):
    external.llm_results = [
        full_extraction().model_copy(update={"location": fact("Paris", "https://invented.example")})
    ]

    lead = await run_enrichment(request())

    assert lead.location == Sourced.empty()
    assert any("dropped 1 fact" in error for error in lead.errors)


async def test_budget_runout_returns_what_was_found(external, monkeypatch, settings):
    """The scrape has finished when the budget runs out, but a search is still hanging."""

    async def search(query, max_results=5):
        await asyncio.sleep(30)

    monkeypatch.setattr(tavily, "search", search)
    settings.enrich_timeout_seconds = 0.2

    lead = await run_enrichment(request())

    assert any(error.startswith("timeout") for error in lead.errors)
    assert lead.domain.value == "acme.com"
    assert [t.value for t in lead.tech_stack] == ["Shopify"]
    assert lead.company_name.value is None
    assert lead.latency_ms < 2000


async def test_budget_runout_during_retry_keeps_first_extraction(external, monkeypatch, settings):
    original = tavily.search
    calls = 0

    async def search(query, max_results=5):
        nonlocal calls
        calls += 1
        if calls == 3:
            await asyncio.sleep(30)
        return await original(query, max_results)

    monkeypatch.setattr(tavily, "search", search)
    settings.enrich_timeout_seconds = 0.2
    external.llm_results = [without_key_fields()]

    lead = await run_enrichment(request())

    assert any(error.startswith("timeout") for error in lead.errors)
    assert lead.company_name.value == "Acme"


async def test_llm_timeout_returns_partial_result(external):
    external.llm_results = [LLMTimeoutError("timed out after 15s")] * 2

    lead = await run_enrichment(request())

    assert "llm: timed out after 15s" in lead.errors
    assert lead.domain.value == "acme.com"
    assert lead.company_name.value is None


async def test_llm_failure_on_retry_keeps_first_extraction(external):
    external.llm_results = [without_key_fields(), LLMError("rate limited after 4 attempts")]

    lead = await run_enrichment(request())

    assert lead.company_name.value == "Acme"
    assert "llm: rate limited after 4 attempts" in lead.errors


async def test_llm_failure_with_no_extraction_raises(external):
    external.llm_results = [LLMError("rate limited after 4 attempts")]

    with pytest.raises(LLMError):
        await run_enrichment(request())
