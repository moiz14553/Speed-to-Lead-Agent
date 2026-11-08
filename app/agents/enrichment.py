import asyncio
import calendar
import re
import time
from datetime import date, timedelta
from operator import add
from typing import Annotated, TypedDict
from urllib.parse import urlsplit

import httpx
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.models.enrichment import (
    Confidence,
    EnrichedLead,
    EnrichRequest,
    ExtractionResult,
    NewsItem,
    Sourced,
    SourceDoc,
)
from app.services import llm, scraper, tavily
from app.services.tech_detect import detect_tech

FREE_EMAIL_RE = re.compile(
    r"^(gmail\.com|googlemail\.com|icloud\.com"
    r"|(yahoo|outlook|hotmail)\.(com|[a-z]{2}|co\.[a-z]{2}|com\.[a-z]{2}))$"
)
DOMAIN_RE = re.compile(r"^([a-z0-9-]+\.)+[a-z]{2,}$")
SEARCH_PROVIDER_HOSTS = (
    "tavily.com",
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "yahoo.com",
    "brave.com",
    "perplexity.ai",
    "exa.ai",
    "serpapi.com",
    "serper.dev",
)
# Company profile pages compiled by a third party. Facts cited from these are low confidence.
AGGREGATOR_HOSTS = SEARCH_PROVIDER_HOSTS + (
    "leadiq.com",
    "zoominfo.com",
    "crunchbase.com",
    "pitchbook.com",
    "cbinsights.com",
    "tracxn.com",
    "rocketreach.co",
    "apollo.io",
    "lusha.com",
    "dnb.com",
    "owler.com",
    "globaldata.com",
    "craft.co",
    "growjo.com",
    "signalhire.com",
    "datanyze.com",
    "theorg.com",
)
# Never the company's own site when looking up a website from a company name.
NON_COMPANY_HOSTS = AGGREGATOR_HOSTS + (
    "linkedin.com",
    "facebook.com",
    "instagram.com",
    "x.com",
    "twitter.com",
    "youtube.com",
    "wikipedia.org",
    "bloomberg.com",
    "glassdoor.com",
    "indeed.com",
    "yelp.com",
)
KEY_FIELDS = ("industry", "employee_range")
TARGETED_QUERY_TERMS = {
    "industry": "what the company does, industry sector",
    "employee_range": "number of employees, company size",
}
SEARCH_ERRORS = (TimeoutError, httpx.HTTPError, tavily.TavilyError, ValueError)


class EnrichState(TypedDict, total=False):
    email: str
    company_hint: str | None
    is_free_email: bool
    stop: bool
    domain: str | None
    domain_source_url: str | None
    domain_confidence: Confidence
    homepage_url: str | None
    target: str
    extraction: ExtractionResult | None
    missing_fields: list[str]
    retries: int
    deadline: float
    llm_gave_up: bool
    sources: Annotated[list[SourceDoc], add]
    tech_stack: Annotated[list[Sourced], add]
    errors: Annotated[list[str], add]
    cost_usd: Annotated[float, add]


# The Annotated fields above, for rebuilding the state from streamed updates.
ACCUMULATED_KEYS = ("sources", "tech_stack", "errors", "cost_usd")


def is_free_email(domain: str) -> bool:
    return bool(FREE_EMAIL_RE.match(domain))


def parse_domain(hint: str) -> str | None:
    """Return the domain if the hint is a domain or URL, None if it is a company name."""
    hint = hint.strip().lower()
    if " " in hint:
        return None
    host = urlsplit(hint if "//" in hint else f"//{hint}").hostname or ""
    host = host.removeprefix("www.")
    return host if DOMAIN_RE.match(host) else None


def _describe_error(exc: Exception, timeout: float) -> str:
    if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
        return f"timed out after {timeout:g}s"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    return str(exc) or type(exc).__name__


def _host_in(host: str, domains: tuple[str, ...]) -> bool:
    return any(host == d or host.endswith(f".{d}") for d in domains)


async def _lookup_website(company: str) -> tuple[str | None, str | None]:
    """Find a company's own website by name. Returns (domain, source URL)."""
    for doc in await tavily.search(f"{company} official website"):
        host = (urlsplit(doc.url).hostname or "").removeprefix("www.")
        if DOMAIN_RE.match(host) and not _host_in(host, NON_COMPANY_HOSTS):
            return host, doc.url
    return None, None


async def resolve_domain(state: EnrichState) -> EnrichState:
    settings = get_settings()
    domain = state["email"].rsplit("@", 1)[1].lower()
    hint = (state.get("company_hint") or "").strip()

    if not is_free_email(domain):
        domain = domain.removeprefix("www.")
        return {
            "is_free_email": False,
            "domain": domain,
            "domain_confidence": "high",
            "target": f"{hint} ({domain})" if hint else domain,
        }

    hint_domain = parse_domain(hint) if hint else None
    if not hint or (hint_domain and is_free_email(hint_domain)):
        return {"is_free_email": True, "stop": True}
    if hint_domain:
        return {
            "is_free_email": True,
            "domain": hint_domain,
            "domain_confidence": "medium",
            "target": hint_domain,
        }

    update: EnrichState = {"is_free_email": True, "target": hint, "domain_confidence": "medium"}
    try:
        found, source_url = await _lookup_website(hint)
    except SEARCH_ERRORS as exc:
        message = _describe_error(exc, settings.tavily_timeout_seconds)
        return {**update, "errors": [f"website lookup: {message}"]}

    update.update(
        domain=found, domain_source_url=source_url, cost_usd=settings.tavily_cost_per_search_usd
    )
    if found:
        update["target"] = f"{hint} ({found})"
    return update


async def scrape_site(state: EnrichState) -> EnrichState:
    domain = state.get("domain")
    if not domain:
        return {"sources": []}

    settings = get_settings()
    pages, errors = await scraper.scrape_site(domain)

    # Tech stack comes straight from the HTML: each match is a fact from that page.
    tech_stack: dict[str, Sourced] = {}
    for page in pages:
        for tech in detect_tech(page.html):
            tech_stack.setdefault(tech, Sourced(value=tech, source_url=page.url, confidence="high"))

    firecrawl_pages = sum(page.via_firecrawl for page in pages)
    return {
        "sources": [SourceDoc(url=page.url, text=page.text) for page in pages],
        "homepage_url": pages[0].url if pages else None,
        "tech_stack": list(tech_stack.values()),
        "errors": errors,
        "cost_usd": firecrawl_pages * settings.firecrawl_cost_per_page_usd,
    }


async def _run_search(label: str, query: str) -> EnrichState:
    settings = get_settings()
    try:
        docs = await tavily.search(query)
    except SEARCH_ERRORS as exc:
        message = _describe_error(exc, settings.tavily_timeout_seconds)
        return {"errors": [f"{label}: {message}"]}
    return {"sources": docs, "cost_usd": settings.tavily_cost_per_search_usd}


async def search_company(state: EnrichState) -> EnrichState:
    query = f"{state['target']} company overview industry employees headquarters"
    return await _run_search("company search", query)


async def search_funding_news(state: EnrichState) -> EnrichState:
    query = f"{state['target']} funding round investors recent news"
    return await _run_search("funding search", query)


def drop_unsourced(
    result: ExtractionResult, sources: list[SourceDoc]
) -> tuple[ExtractionResult, int]:
    """Null every fact whose source_url is not one we collected. Returns the count dropped."""
    allowed_urls = {doc.url.rstrip("/") for doc in sources}

    def is_sourced(fact: Sourced) -> bool:
        return (fact.source_url or "").rstrip("/") in allowed_urls

    dropped = 0
    update: dict = {}
    for field in ExtractionResult.model_fields:
        value = getattr(result, field)
        if isinstance(value, list):
            found = [fact for fact in value if fact.value is not None]
            update[field] = [fact for fact in found if is_sourced(fact)]
            dropped += len(found) - len(update[field])
        elif value.value is not None and not is_sourced(value):
            update[field] = Sourced.empty()
            dropped += 1
    return result.model_copy(update=update), dropped


def _capped_confidence(fact: Sourced, own_hosts: tuple[str, ...]) -> Confidence:
    """High needs the company's own site, third parties reach medium, aggregators are low."""
    host = urlsplit(fact.source_url or "").hostname or ""
    if _host_in(host, own_hosts):
        return fact.confidence or "low"
    if _host_in(host, AGGREGATOR_HOSTS) or fact.confidence in (None, "low"):
        return "low"
    return "medium"


def apply_confidence_rules(
    result: ExtractionResult, own_hosts: tuple[str, ...]
) -> ExtractionResult:
    def capped(fact: Sourced) -> Sourced:
        if fact.value is None:
            return fact
        return fact.model_copy(update={"confidence": _capped_confidence(fact, own_hosts)})

    update: dict = {}
    for field in ExtractionResult.model_fields:
        value = getattr(result, field)
        update[field] = [capped(fact) for fact in value] if isinstance(value, list) else capped(value)
    return result.model_copy(update=update)


def _date_span(text: str) -> tuple[date, date] | None:
    """Return the first and last day a YYYY-MM-DD or YYYY-MM date could mean."""
    try:
        if re.fullmatch(r"\d{4}-\d{2}", text):
            first = date.fromisoformat(f"{text}-01")
            last_day = calendar.monthrange(first.year, first.month)[1]
            return first, first.replace(day=last_day)
        day = date.fromisoformat(text)
    except ValueError:
        return None
    return day, day


def is_recent_news(item: NewsItem, today: date) -> bool:
    """Keep only items dated within the last 12 months. No usable date means it can't be shown."""
    span = _date_span(item.published_date or "")
    if span is None:
        return False
    first, last = span
    return last >= today - timedelta(days=365) and first <= today


def _own_hosts(state: EnrichState) -> tuple[str, ...]:
    urls = [state.get("homepage_url")]
    hosts = [(urlsplit(url).hostname or "").removeprefix("www.") for url in urls if url]
    return tuple(dict.fromkeys([state["domain"], *hosts] if state.get("domain") else hosts))


def _merge_extractions(
    previous: ExtractionResult | None, current: ExtractionResult
) -> ExtractionResult:
    """Keep facts from the first extraction that the second one came back without."""
    if previous is None:
        return current
    update = {}
    for field in ExtractionResult.model_fields:
        value = getattr(current, field)
        if not (value if isinstance(value, list) else value.value):
            update[field] = getattr(previous, field)
    return current.model_copy(update=update)


async def extract(state: EnrichState) -> EnrichState:
    sources = state.get("sources") or []
    if not sources:
        return {"errors": ["extract: skipped, no sources collected"]}

    previous = state.get("extraction")
    try:
        result, cost, warning = await llm.extract_facts(
            state["target"], sources, state.get("deadline")
        )
    except llm.LLMWaitTooLongError as exc:
        return {"errors": [f"llm: {exc}"], "llm_gave_up": True}
    except llm.LLMTimeoutError as exc:
        return {"errors": [f"llm: {exc}"]}
    except llm.LLMError as exc:
        if previous is None:
            raise
        return {"errors": [f"llm: {exc}"]}

    result, dropped = drop_unsourced(result, sources)
    result = apply_confidence_rules(result, _own_hosts(state))
    today = date.today()
    recent = [item for item in result.recent_news if is_recent_news(item, today)]
    result = result.model_copy(update={"recent_news": recent})
    errors = [warning] if warning else []
    if dropped:
        errors.append(f"validate: dropped {dropped} fact(s) citing a URL that was not collected")
    return {
        "extraction": _merge_extractions(previous, result),
        "cost_usd": cost,
        "errors": errors,
    }


async def validate(state: EnrichState) -> EnrichState:
    extraction = state.get("extraction")
    return {
        "missing_fields": [
            field
            for field in KEY_FIELDS
            if extraction is None or getattr(extraction, field).value is None
        ]
    }


async def targeted_search(state: EnrichState) -> EnrichState:
    terms = ", ".join(TARGETED_QUERY_TERMS[field] for field in state["missing_fields"])
    update = await _run_search("targeted search", f"{state['target']} {terms}")
    return {**update, "retries": state.get("retries", 0) + 1}


def _route_after_resolve(state: EnrichState) -> list[str] | str:
    if state.get("stop"):
        return END
    return ["scrape_site", "search_company", "search_funding_news"]


def _route_after_validate(state: EnrichState) -> str:
    if state.get("llm_gave_up"):
        return END
    if state["missing_fields"] and state.get("retries", 0) < 1:
        return "targeted_search"
    return END


def build_graph():
    graph = StateGraph(EnrichState)
    graph.add_node("resolve_domain", resolve_domain)
    graph.add_node("scrape_site", scrape_site)
    graph.add_node("search_company", search_company)
    graph.add_node("search_funding_news", search_funding_news)
    graph.add_node("extract", extract)
    graph.add_node("validate", validate)
    graph.add_node("targeted_search", targeted_search)

    graph.add_edge(START, "resolve_domain")
    graph.add_conditional_edges(
        "resolve_domain",
        _route_after_resolve,
        ["scrape_site", "search_company", "search_funding_news", END],
    )
    graph.add_edge(["scrape_site", "search_company", "search_funding_news"], "extract")
    graph.add_edge("extract", "validate")
    graph.add_conditional_edges("validate", _route_after_validate, ["targeted_search", END])
    graph.add_edge("targeted_search", "extract")
    return graph.compile()


enrichment_graph = build_graph()


def _apply_update(state: EnrichState, update: EnrichState) -> None:
    for key, value in update.items():
        state[key] = state[key] + value if key in ACCUMULATED_KEYS else value


def _domain_fact(state: EnrichState) -> Sourced:
    if not state.get("domain"):
        return Sourced.empty()
    source_url = state.get("homepage_url") or state.get("domain_source_url")
    return Sourced(
        value=state["domain"],
        source_url=source_url,
        confidence=state["domain_confidence"] if source_url else "low",
    )


def _build_lead(state: EnrichState, is_free: bool, latency_ms: int) -> EnrichedLead:
    settings = get_settings()
    extraction = state.get("extraction")
    return EnrichedLead(
        **(extraction.model_dump() if extraction else {}),
        domain=_domain_fact(state),
        tech_stack=state["tech_stack"],
        is_free_email=is_free,
        cost_usd=round(state["cost_usd"], 6),
        latency_ms=latency_ms,
        llm_provider=settings.llm_provider,
        llm_model=settings.resolved_model,
        errors=list(dict.fromkeys(state["errors"])),
    )


async def run_enrichment(request: EnrichRequest) -> EnrichedLead:
    """Run the graph within the time budget. On timeout, return what was found so far."""
    settings = get_settings()
    start = time.perf_counter()
    state: EnrichState = {
        "email": request.email,
        "company_hint": request.company_hint,
        "retries": 0,
        "deadline": time.monotonic() + settings.enrich_timeout_seconds,
        "sources": [],
        "tech_stack": [],
        "errors": [],
        "cost_usd": 0.0,
    }

    # Stream node updates rather than waiting for the final state, so results
    # from finished nodes survive if the budget cuts the run short.
    try:
        async with asyncio.timeout(settings.enrich_timeout_seconds):
            async for chunk in enrichment_graph.astream(state, stream_mode="updates"):
                for update in chunk.values():
                    _apply_update(state, update or {})
    except TimeoutError:
        state["errors"] = state["errors"] + [
            f"timeout: stopped at the {settings.enrich_timeout_seconds:g}s budget, "
            "results are partial"
        ]

    email_domain = request.email.rsplit("@", 1)[1].lower()
    is_free = state.get("is_free_email", is_free_email(email_domain))
    latency_ms = int((time.perf_counter() - start) * 1000)
    return _build_lead(state, is_free, latency_ms)
