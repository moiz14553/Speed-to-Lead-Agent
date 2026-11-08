import asyncio
import json
from datetime import date, timedelta

import httpx
import pytest
from pydantic import ValidationError

from app.agents.enrichment import (
    domain_matches_company,
    drop_unsourced,
    employee_bucket,
    is_free_email,
    is_recent_news,
    parse_domain,
    run_enrichment,
)
from app.models.enrichment import EnrichRequest, OwnershipFact, Sourced, SourceDoc
from app.services import scraper, tavily
from app.services.llm import LLMError, LLMTimeoutError, LLMWaitTooLongError
from tests.conftest import HOMEPAGE, SEARCH_URL, fact, full_extraction, news


def request(email="jo@acme.com", hint=None) -> EnrichRequest:
    return EnrichRequest(email=email, name="Jo", company_hint=hint)


def without_key_fields():
    return full_extraction().model_copy(
        update={
            "own_site_industry": Sourced.empty(),
            "third_party_industry": Sourced.empty(),
            "employee_count": Sourced.empty(),
        }
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


@pytest.mark.parametrize(
    "not_a_website",
    [
        "https://tavily.com/knowledge/company/abc",
        "https://www.tavily.com/",
        "https://leadiq.com/c/acme/123",
        "https://www.cbinsights.com/company/acme",
    ],
)
async def test_lookup_never_uses_search_or_aggregator_pages(external, monkeypatch, not_a_website):
    async def search(query, max_results=5):
        return [
            SourceDoc(url=not_a_website, text="Acme profile"),
            SourceDoc(url="https://acme.com/", text="Acme home"),
        ]

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request("jo@gmail.com", "Acme Inc"))

    assert lead.domain.value == "acme.com"


async def test_lookup_with_only_search_pages_finds_no_domain(external, monkeypatch):
    async def search(query, max_results=5):
        return [SourceDoc(url="https://tavily.com/knowledge/company/abc", text="Acme profile")]

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request("jo@gmail.com", "Acme Inc"))

    assert lead.domain.value is None
    assert external.scraped_domains == []


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


async def test_news_has_its_own_search(external):
    await run_enrichment(request())

    assert external.news_queries == ["acme.com news"]
    funding_query = next(q for q in external.search_queries if "funding" in q)
    assert "news" not in funding_query


async def test_news_search_results_reach_the_llm(external, monkeypatch):
    async def search_news(query, max_results=5):
        return [
            SourceDoc(
                url="https://reuters.com/acme",
                title="Acme expands",
                published_date="2026-10-01",
                text="Acme opened a plant.",
            )
        ]

    monkeypatch.setattr(tavily, "search_news", search_news)

    await run_enrichment(request())

    doc = next(d for d in external.llm_sources[0] if d.url == "https://reuters.com/acme")
    assert doc.published_date == "2026-10-01"


async def test_failed_news_search_is_reported_and_run_continues(external, monkeypatch):
    async def search_news(query, max_results=5):
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(tavily, "search_news", search_news)

    lead = await run_enrichment(request())

    assert "news search: boom" in lead.errors
    assert lead.company_name.value == "Acme"


def _sources_for(*docs):
    async def search(query, max_results=5):
        return list(docs)

    return search


async def test_small_company_keeps_only_results_about_it(external, monkeypatch):
    """Searches for a tiny company mostly come back about other companies."""
    profile = SourceDoc(url="https://tavily.com/knowledge/company/x", text="Soapplai builds agents.")
    unrelated_funding = SourceDoc(
        url="https://multiples.vc/private-comps/approaching-ai",
        text="Approaching.AI raised a $195M Series B in October 2026.",
    )
    unrelated_profile = SourceDoc(
        url="https://www.cbinsights.com/company/edgar-dunn", text="Edgar Dunn raised a Series B."
    )
    monkeypatch.setattr(tavily, "search", _sources_for(profile, unrelated_funding, unrelated_profile))

    await run_enrichment(request("info@soapplai.com"))

    urls = {doc.url for doc in external.llm_sources[0]}
    assert profile.url in urls
    assert unrelated_funding.url not in urls
    assert unrelated_profile.url not in urls


async def test_llm_cannot_cite_a_dropped_source(external, monkeypatch):
    unrelated = SourceDoc(url="https://other.example/x", text="Other Corp raised $5M.")
    monkeypatch.setattr(tavily, "search", _sources_for(unrelated))
    external.llm_results = [
        full_extraction().model_copy(update={"funding": fact("$5M", "https://other.example/x")})
    ]

    lead = await run_enrichment(request())

    assert lead.funding.value is None
    assert any("citing a URL that was not collected" in error for error in lead.errors)


async def test_llm_is_skipped_when_nothing_is_about_the_company(external, monkeypatch):
    async def scrape(domain):
        return [], ["scrape /: HTTP 403"]

    monkeypatch.setattr(scraper, "scrape_site", scrape)
    monkeypatch.setattr(tavily, "search", _sources_for(SourceDoc(url="https://x.example", text="Other")))

    lead = await run_enrichment(request())

    assert external.llm_calls == 0
    assert "extract: skipped, no sources about this company" in lead.errors


async def test_site_name_counts_as_a_name_match(external, monkeypatch):
    """systemsltd.com doesn't contain the words 'Systems Limited', but its og:site_name does."""
    html = '<meta property="og:site_name" content="Systems Limited">'

    async def scrape(domain):
        return [scraper.Page(url="https://systemsltd.com/", html=html, text="Home")], []

    doc = SourceDoc(url="https://pk.linkedin.com/company/x", text="Systems Limited is an IT firm.")
    generic = SourceDoc(url="https://x.example", text="Our operating systems are limited to two.")
    monkeypatch.setattr(scraper, "scrape_site", scrape)
    monkeypatch.setattr(tavily, "search", _sources_for(doc, generic))

    await run_enrichment(request("info@systemsltd.com"))

    urls = {source.url for source in external.llm_sources[0]}
    assert doc.url in urls
    assert generic.url not in urls


@pytest.mark.parametrize(
    "email,hint,text,expected",
    [
        ("info@kyliecosmetics.com", None, "Kylie Cosmetics launched a palette.", True),
        ("info@acme.com", None, "See acme.com for details.", True),
        ("info@acme.com", None, "ACME widgets are great.", True),
        ("info@acme.com", None, "Acme-branded widgets.", True),
        ("info@stripe.com", None, "A pinstripe suit.", False),
        ("info@acme.com", None, "An unrelated page.", False),
        ("jo@gmail.com", "Stripe, Inc.", "Stripe launched a product.", True),
        ("jo@acme.com", "Systems Limited", "Operating systems news.", False),
    ],
)
async def test_relevance_matching(external, monkeypatch, email, hint, text, expected):
    doc = SourceDoc(url="https://third.example/page", text=text)
    monkeypatch.setattr(tavily, "search", _sources_for(doc))

    await run_enrichment(request(email, hint))

    assert any(s.url == doc.url for s in sum(external.llm_sources, [])) is expected


async def test_own_site_is_always_kept(external):
    await run_enrichment(request("info@acme.com"))

    assert any(doc.url == HOMEPAGE for doc in external.llm_sources[0])


async def test_debug_save_writes_collected_data(external, settings, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings.debug_save_raw = True
    unrelated = SourceDoc(url="https://other.example/x", text="Other Corp.")
    related = SourceDoc(url="https://news.example/acme", text="Acme news.")
    monkeypatch.setattr(tavily, "search", _sources_for(unrelated, related))

    await run_enrichment(request(), "run1")

    data = json.loads((tmp_path / "debug" / "run1" / "jo@acme.com.json").read_text(encoding="utf-8"))
    kept = {source["url"]: source["kept_by_relevance_filter"] for source in data["sources"]}
    assert kept["https://other.example/x"] is False
    assert kept["https://news.example/acme"] is True
    assert "cdn.shopify.com" in data["raw_html"][HOMEPAGE]
    assert data["extraction"]["company_name"]["value"] == "Acme"


async def test_debug_save_includes_website_lookup_results(
    external, settings, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    settings.debug_save_raw = True
    monkeypatch.setattr(
        tavily, "search", _sources_for(SourceDoc(url="https://financer.com/stripe", text="Review"))
    )

    await run_enrichment(request("jo@gmail.com", "Stripe"), "run1")

    data = json.loads((tmp_path / "debug" / "run1" / "jo@gmail.com.json").read_text(encoding="utf-8"))
    assert data["website_lookup"][0]["url"] == "https://financer.com/stripe"
    assert data["domain"] is None


async def test_debug_save_is_off_by_default(external, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    await run_enrichment(request(), "run1")

    assert not (tmp_path / "debug").exists()


async def test_debug_save_also_runs_when_llm_fails(external, settings, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings.debug_save_raw = True
    external.llm_results = [LLMError("rate limited after 4 attempts")]

    with pytest.raises(LLMError):
        await run_enrichment(request(), "run1")

    assert (tmp_path / "debug" / "run1" / "jo@acme.com.json").exists()


@pytest.mark.parametrize("run_id", ["../../evil", "..", "a/b\\c"])
async def test_debug_folder_cannot_escape_debug_dir(
    external, settings, monkeypatch, tmp_path, run_id
):
    monkeypatch.chdir(tmp_path)
    settings.debug_save_raw = True

    await run_enrichment(request(), run_id)

    debug_root = (tmp_path / "debug").resolve()
    files = list(tmp_path.rglob("jo@acme.com.json"))
    assert len(files) == 1
    assert files[0].resolve().is_relative_to(debug_root)


async def test_debug_save_failure_does_not_fail_the_run(external, settings, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings.debug_save_raw = True
    (tmp_path / "debug").write_text("a file where the folder should go", encoding="utf-8")

    lead = await run_enrichment(request(), "run1")

    assert lead.company_name.value == "Acme"


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
        update={"location": [], "recent_news": []}
    )
    external.llm_results = [without_key_fields(), second]

    lead = await run_enrichment(request())

    assert lead.location[0].value == "Toronto"
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
            "own_site_industry": fact("Made up", "https://invented.example/"),
            "recent_news": [
                news("Real", SEARCH_URL),
                news("Fake", "https://invented.example/news"),
            ],
        }
    )
    sources = [SourceDoc(url=HOMEPAGE, text="x"), SourceDoc(url=SEARCH_URL, text="y")]

    cleaned, dropped = drop_unsourced(extraction, sources)

    assert dropped == 2
    assert cleaned.own_site_industry == Sourced.empty()
    assert [item.value for item in cleaned.recent_news] == ["Real"]
    assert cleaned.company_name.value == "Acme"


async def test_invented_source_is_nulled_and_reported(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={"location": [fact("Paris", "https://invented.example")]}
        )
    ]

    lead = await run_enrichment(request())

    assert lead.location == []
    assert any("dropped 1 fact" in error for error in lead.errors)


@pytest.mark.parametrize(
    "url,given,expected",
    [
        ("https://acme.com/", "high", "high"),
        ("https://blog.acme.com/post", "high", "high"),
        ("https://acme.com/", "medium", "medium"),
        ("https://en.wikipedia.org/wiki/Acme", "high", "medium"),
        ("https://pk.linkedin.com/company/acme", "high", "medium"),
        ("https://www.crunchbase.com/organization/acme", "high", "medium"),
        ("https://www.reuters.com/business/acme", "high", "medium"),
        ("https://www.dawn.com/news/1", "medium", "medium"),
        ("https://en.wikipedia.org/wiki/Acme", "low", "low"),
        ("https://tavily.com/knowledge/company/abc", "high", "low"),
        ("https://leadiq.com/c/acme/1", "medium", "low"),
        ("https://enrichlayer.com/company/acme", "high", "low"),
        ("https://notacme.com/", "high", "low"),
        ("https://wikipedia.org.evil.com/acme", "high", "low"),
    ],
)
async def test_confidence_depends_on_the_source(external, monkeypatch, url, given, expected):
    async def search(query, max_results=5):
        return [SourceDoc(url=url, text="About Acme")]

    monkeypatch.setattr(tavily, "search", search)
    external.llm_results = [
        full_extraction().model_copy(
            update={"location": [Sourced(value="Toronto", source_url=url, confidence=given)]}
        )
    ]

    lead = await run_enrichment(request())

    assert lead.location[0].confidence == expected


async def test_news_confidence_is_capped_too(external):
    lead = await run_enrichment(request())

    assert lead.recent_news[0].confidence == "low"


async def test_low_confidence_for_disagreeing_sources_is_kept_on_own_site(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={
                "own_site_industry": Sourced(value="Widgets", source_url=HOMEPAGE, confidence="low")
            }
        )
    ]

    lead = await run_enrichment(request())

    assert lead.industry.confidence == "low"


async def test_several_offices_are_kept_when_own_site_lists_them(external):
    external.llm_results = [
        full_extraction().model_copy(update={"location": [fact("Milton"), fact("Karachi")]})
    ]

    lead = await run_enrichment(request())

    assert [office.value for office in lead.location] == ["Milton", "Karachi"]
    assert all(office.confidence == "high" for office in lead.location)


async def test_at_most_three_locations(external):
    offices = [fact(city) for city in ("Toronto", "Karachi", "Lahore", "Dubai", "London")]
    external.llm_results = [full_extraction().model_copy(update={"location": offices})]

    lead = await run_enrichment(request())

    assert [office.value for office in lead.location] == ["Toronto", "Karachi", "Lahore"]


@pytest.mark.parametrize(
    "report_url",
    [
        "https://acme.com/files/Annual%20Report%202025.pdf",
        "https://acme.com/files/report.pdf",
        "https://acme.com/investors/annual-report/offices",
    ],
)
async def test_annual_report_branch_lists_are_ignored(external, monkeypatch, report_url):
    async def search(query, max_results=5):
        return [SourceDoc(url=report_url, text="Branch offices")]

    monkeypatch.setattr(tavily, "search", search)
    external.llm_results = [
        full_extraction().model_copy(
            update={
                "location": [fact("Multan branch", report_url), fact("Toronto"), fact("Lahore")]
            }
        )
    ]

    lead = await run_enrichment(request())

    assert [office.value for office in lead.location] == ["Toronto", "Lahore"]


async def test_own_site_location_preferred_over_third_party(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={"location": [fact("Wrong City", SEARCH_URL), fact("Toronto")]}
        )
    ]

    lead = await run_enrichment(request())

    assert [office.value for office in lead.location] == ["Toronto"]


async def test_only_first_location_kept_when_not_from_own_site(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={"location": [fact("Milton", SEARCH_URL), fact("Karachi", SEARCH_URL)]}
        )
    ]

    lead = await run_enrichment(request())

    assert [office.value for office in lead.location] == ["Milton"]


@pytest.mark.parametrize(
    "host,company,expected",
    [
        ("stripe.com", "Stripe", True),
        ("stripe.com", "Stripe, Inc.", True),
        ("docs.stripe.com", "Stripe", True),
        ("acme.co.uk", "Acme Ltd", True),
        ("bankofamerica.com", "Bank of America", True),
        ("daraz.pk", "Daraz Pakistan", True),
        ("systemsltd.com", "Systems Limited", True),
        ("ibm.com", "International Business Machines", True),
        ("financer.com", "Stripe", False),
        ("stripe.financer.com", "Stripe", False),
        ("linkedin.com", "Stripe", False),
        ("shopper.com", "HP", False),
        ("example.com", "The Company", False),
    ],
)
def test_domain_matches_company(host, company, expected):
    assert domain_matches_company(host, company) is expected


async def test_lookup_finds_exact_match_past_other_results(external, monkeypatch):
    async def search(query, max_results=5):
        if "official website" not in query:
            return []
        assert max_results == 10
        return [
            SourceDoc(url="https://tavily.com/knowledge/company/abc", text="Stripe"),
            SourceDoc(url="https://financer.com/stripe", text="Stripe review"),
            SourceDoc(url="https://stripepayments.com/", text="Stripe"),
            SourceDoc(url="https://stripe.dev/", text="Stripe developers"),
            SourceDoc(url="https://docs.stripe.com/", text="Stripe docs"),
            SourceDoc(url="https://stripe.com/", text="Stripe"),
        ]

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request("jo@gmail.com", "Stripe"))

    assert lead.domain.value == "stripe.com"
    assert external.scraped_domains == ["stripe.com"]


async def test_lookup_falls_back_to_a_partial_name_match(external, monkeypatch):
    async def search(query, max_results=5):
        return [SourceDoc(url="https://acmewidgets.io/", text="Acme")]

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request("jo@gmail.com", "Acme"))

    assert lead.domain.value == "acmewidgets.io"


async def test_lookup_returns_null_when_no_site_matches_the_name(external, monkeypatch):
    async def search(query, max_results=5):
        return [SourceDoc(url="https://financer.com/stripe", text="Stripe review")]

    monkeypatch.setattr(tavily, "search", search)

    lead = await run_enrichment(request("jo@gmail.com", "Stripe"))

    assert lead.domain.value is None
    assert external.scraped_domains == []


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("120", "51-200"),
        ("120 employees", "51-200"),
        ("5", "1-10"),
        ("10", "1-10"),
        ("11", "11-50"),
        ("50", "11-50"),
        ("51", "51-200"),
        ("200", "51-200"),
        ("201", "201-500"),
        ("500", "201-500"),
        ("501", "501-1000"),
        ("1000", "501-1000"),
        ("1001", "1001-5000"),
        ("5000", "1001-5000"),
        ("5001", "5001-10000"),
        ("10000", "5001-10000"),
        ("10001", "10000+"),
        ("13,651", "10000+"),
        ("228,000 employees", "10000+"),
        ("about 75 people", "51-200"),
        ("2.5k", "1001-5000"),
        ("1.2 million", "10000+"),
        ("51-200", "51-200"),
        ("1,001-5,000 employees", "1001-5000"),
        ("5,001 to 10,000", "5001-10000"),
        ("201 \u2013 500", "201-500"),
        ("1k-5k", "1001-5000"),
        ("5-10k", "5001-10000"),
        ("10,001+", "10000+"),
        ("10000+", "10000+"),
        ("10,000+ employees", "10000+"),
        ("200+", "201-500"),
        ("over 200", "201-500"),
        ("View all 158,097 employees on LinkedIn", None),
        ("158k followers", None),
        ("unknown", None),
        ("hundreds", None),
        ("0", None),
        ("", None),
        (None, None),
    ],
)
def test_employee_bucket_mapping(raw, expected):
    assert employee_bucket(raw) == expected


async def test_llm_headcount_is_mapped_to_a_bucket_with_its_source(external):
    external.llm_results = [
        full_extraction().model_copy(update={"employee_count": fact("1,500 staff", SEARCH_URL)})
    ]

    lead = await run_enrichment(request())

    assert lead.employee_range.value == "1001-5000"
    assert lead.employee_range.source_url == SEARCH_URL


async def test_unmappable_headcount_is_null_and_triggers_the_retry(external):
    external.llm_results = [
        full_extraction().model_copy(update={"employee_count": fact("a lot of people")}),
        full_extraction(),
    ]

    lead = await run_enrichment(request())

    assert external.llm_calls == 2
    assert lead.employee_range.value == "51-200"


def _ownership(value, acquired_by=None, url=SEARCH_URL):
    return OwnershipFact(value=value, acquired_by=acquired_by, source_url=url, confidence="medium")


@pytest.mark.parametrize(
    "value,acquired_by,expected",
    [
        ("private", None, "private"),
        ("public", None, "public"),
        ("acquired", "Coty", "acquired_by: Coty"),
        ("acquired", None, "acquired_by: unknown"),
    ],
)
async def test_ownership_field(external, value, acquired_by, expected):
    external.llm_results = [
        full_extraction().model_copy(update={"ownership": _ownership(value, acquired_by)})
    ]

    lead = await run_enrichment(request())

    assert lead.ownership.value == expected
    assert lead.ownership.source_url == SEARCH_URL


@pytest.mark.parametrize("value", ["public", "acquired"])
async def test_public_or_acquired_clears_funding(external, value):
    external.llm_results = [
        full_extraction().model_copy(update={"ownership": _ownership(value, "Alibaba")})
    ]

    lead = await run_enrichment(request())

    assert lead.funding.value is None


@pytest.mark.parametrize("value", ["private", None])
async def test_funding_kept_unless_public_or_acquired(external, value):
    external.llm_results = [
        full_extraction().model_copy(update={"ownership": _ownership(value)})
    ]

    lead = await run_enrichment(request())

    assert lead.funding.value == "Series A"


async def test_funding_cleared_when_only_the_first_extraction_found_ownership(external):
    first = full_extraction().model_copy(
        update={
            "ownership": _ownership("public"),
            "own_site_industry": Sourced.empty(),
            "funding": Sourced.empty(),
        }
    )
    second = full_extraction().model_copy(update={"ownership": _ownership(None)})
    external.llm_results = [first, second]

    lead = await run_enrichment(request())

    assert lead.ownership.value == "public"
    assert lead.funding.value is None


@pytest.mark.parametrize(
    "summary,expected",
    [
        ("Acme makes widgets. It was acquired by Coty in 2019.", "Acme makes widgets."),
        ("Acme makes widgets. Note: The company is publicly traded.", "Acme makes widgets."),
        ("Acme makes widgets. Listed on NASDAQ (ACME).", "Acme makes widgets."),
        ("Acme makes widgets. It sells them online.", "Acme makes widgets. It sells them online."),
        ("Acme is a publicly traded company.", None),
    ],
)
async def test_ownership_is_removed_from_summary(external, summary, expected):
    external.llm_results = [
        full_extraction().model_copy(update={"website_summary": fact(summary)})
    ]

    lead = await run_enrichment(request())

    assert lead.website_summary.value == expected


async def test_own_site_industry_beats_third_party_label(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={
                "own_site_industry": fact("Online marketplace"),
                "third_party_industry": fact("Software Development", "https://www.linkedin.com/x"),
            }
        )
    ]

    lead = await run_enrichment(request())

    assert lead.industry.value == "Online marketplace"


async def test_third_party_industry_used_when_own_site_has_none(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={
                "own_site_industry": Sourced.empty(),
                "third_party_industry": fact("Retail", SEARCH_URL),
            }
        )
    ]

    lead = await run_enrichment(request())

    assert lead.industry.value == "Retail"


async def test_own_site_industry_from_another_site_is_not_treated_as_own(external):
    external.llm_results = [
        full_extraction().model_copy(
            update={
                "own_site_industry": fact("Claimed", SEARCH_URL),
                "third_party_industry": fact("Other label", SEARCH_URL),
            }
        )
    ]

    lead = await run_enrichment(request())

    assert lead.industry.value == "Other label"


TODAY = date(2026, 11, 8)


@pytest.mark.parametrize(
    "published,expected",
    [
        ("2026-11-08", True),
        ("2026-01-15", True),
        ("2025-11-08", True),
        ("2025-11", True),
        ("2025-11-07", False),
        ("2025-10", False),
        ("2019-11-01", False),
        ("2026-11-09", False),
        ("2026-12", False),
        ("2026", False),
        ("October", False),
        (None, False),
    ],
)
def test_news_must_be_dated_within_12_months(published, expected):
    item = news("Item").model_copy(update={"published_date": published})

    assert is_recent_news(item, TODAY) is expected


async def test_old_and_undated_news_is_removed(external):
    old = (date.today() - timedelta(days=500)).isoformat()
    undated = news("Undated", SEARCH_URL).model_copy(update={"published_date": None})
    external.llm_results = [
        full_extraction().model_copy(
            update={"recent_news": [news("Old", SEARCH_URL, old), undated, news("New", SEARCH_URL)]}
        )
    ]

    lead = await run_enrichment(request())

    assert [item.value for item in lead.recent_news] == ["New"]
    assert lead.recent_news[0].published_date == date.today().isoformat()


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


async def test_long_provider_wait_records_error_and_stops(external):
    message = "gave up, provider asked to wait 70919s with 24s left in the time budget"
    external.llm_results = [LLMWaitTooLongError(message)]

    lead = await run_enrichment(request())

    assert f"llm: {message}" in lead.errors
    assert lead.domain.value == "acme.com"
    assert [t.value for t in lead.tech_stack] == ["Shopify"]
    # No targeted search and no second LLM call, which would only use up more quota.
    assert external.llm_calls == 1
    assert len(external.search_queries) == 2


async def test_llm_failure_on_retry_keeps_first_extraction(external):
    external.llm_results = [without_key_fields(), LLMError("rate limited after 4 attempts")]

    lead = await run_enrichment(request())

    assert lead.company_name.value == "Acme"
    assert "llm: rate limited after 4 attempts" in lead.errors


async def test_llm_failure_with_no_extraction_raises(external):
    external.llm_results = [LLMError("rate limited after 4 attempts")]

    with pytest.raises(LLMError):
        await run_enrichment(request())
