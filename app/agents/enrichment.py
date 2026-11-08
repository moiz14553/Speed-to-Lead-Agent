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
    OwnershipFact,
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
# The only third parties that can reach medium confidence. Any other site is low.
TRUSTED_THIRD_PARTY_HOSTS = (
    "linkedin.com",
    "wikipedia.org",
    "crunchbase.com",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "nytimes.com",
    "ft.com",
    "forbes.com",
    "techcrunch.com",
    "cnbc.com",
    "bbc.com",
    "bbc.co.uk",
    "apnews.com",
    "economist.com",
    "theguardian.com",
    "washingtonpost.com",
    "fortune.com",
    "businessinsider.com",
    "dawn.com",
    "tribune.com.pk",
    "thenews.com.pk",
    "brecorder.com",
)
COMPANY_NAME_NOISE = {
    "the",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "gmbh",
    "pvt",
    "pte",
    "group",
    "holdings",
}
EMPLOYEE_BUCKETS = (
    (10, "1-10"),
    (50, "11-50"),
    (200, "51-200"),
    (500, "201-500"),
    (1000, "501-1000"),
    (5000, "1001-5000"),
    (10000, "5001-10000"),
)
_COUNT = r"(\d[\d,]*(?:\.\d+)?)\s*(?:(k|m|thousand|million)\b)?"
_COUNT_RANGE_RE = re.compile(rf"{_COUNT}\s*(?:-|\u2013|\u2014|to)\s*{_COUNT}")
_COUNT_RE = re.compile(_COUNT)
_ABOVE_RE = re.compile(r"\b(over|more than|above)\b")
_PROFILE_COUNT_RE = re.compile(r"follower|profile|on linkedin|linkedin member|associated member")
_UNIT_MULTIPLIERS = {"k": 1_000, "thousand": 1_000, "m": 1_000_000, "million": 1_000_000}
_OWNERSHIP_TEXT_RE = re.compile(
    r"publicly[- ](traded|listed|held)|public company|privately[- ](held|owned)|private company"
    r"|acquired by|was acquired|acquisition by|subsidiary of|owned by|nasdaq|nyse|stock exchange",
    re.IGNORECASE,
)
_REPORT_URL_RE = re.compile(r"\.pdf($|\?)|annual[-_ ]?(%20)?report")
MAX_LOCATIONS = 3
LOOKUP_MAX_RESULTS = 10
DOMAIN_SUFFIX_LABELS = {"com", "co", "org", "net", "io", "ai", "app", "biz", "info"}
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


def _site_label(host: str) -> str:
    """Return the name part of a host: 'stripe' for docs.stripe.co.uk."""
    labels = host.split(".")
    while len(labels) > 1 and (labels[-1] in DOMAIN_SUFFIX_LABELS or len(labels[-1]) == 2):
        labels.pop()
    return labels[-1].replace("-", "")


def _company_tokens(company: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", company.lower()) if t not in COMPANY_NAME_NOISE]


def domain_matches_company(host: str, company: str) -> bool:
    tokens = _company_tokens(company)
    if not tokens:
        return False
    label = _site_label(host)
    name = "".join(tokens)
    return (
        label == name
        or (len(name) >= 3 and name in label)
        or (len(label) >= 4 and name.startswith(label))
        or (len(tokens) >= 2 and label == "".join(token[0] for token in tokens))
    )


async def _lookup_website(company: str) -> tuple[str | None, str | None]:
    """Find a company's own website by name. Returns (domain, source URL), or nulls if none fits."""
    best: tuple[tuple[bool, bool, int], str, str] | None = None
    name = "".join(_company_tokens(company))
    for doc in await tavily.search(f"{company} official website", max_results=LOOKUP_MAX_RESULTS):
        host = (urlsplit(doc.url).hostname or "").removeprefix("www.")
        if (
            not DOMAIN_RE.match(host)
            or _host_in(host, SEARCH_PROVIDER_HOSTS)
            or not domain_matches_company(host, company)
        ):
            continue
        # stripe.com beats stripe.dev, which beats stripepayments.com. First result wins ties.
        score = (_site_label(host) == name, host.endswith(".com"), -host.count("."))
        if best is None or score > best[0]:
            best = (score, host, doc.url)
    return (best[1], best[2]) if best else (None, None)


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
            update[field] = value.model_copy(
                update={"value": None, "source_url": None, "confidence": None}
            )
            dropped += 1
    return result.model_copy(update=update), dropped


def _source_host(fact: Sourced) -> str:
    return urlsplit(fact.source_url or "").hostname or ""


def _capped_confidence(fact: Sourced, own_hosts: tuple[str, ...]) -> Confidence:
    """High needs the company's own site, trusted third parties reach medium, the rest are low."""
    host = _source_host(fact)
    if _host_in(host, own_hosts):
        return fact.confidence or "low"
    if _host_in(host, TRUSTED_THIRD_PARTY_HOSTS) and fact.confidence in ("high", "medium"):
        return "medium"
    return "low"


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


def _pick_locations(locations: list[Sourced], own_hosts: tuple[str, ...]) -> list[Sourced]:
    """Use the company's own site if it gives any, otherwise only the first location."""
    usable = [f for f in locations if not _REPORT_URL_RE.search((f.source_url or "").lower())]
    own = [fact for fact in usable if _host_in(_source_host(fact), own_hosts)]
    return own[:MAX_LOCATIONS] if own else usable[:1]


def _count_value(number: str, unit: str | None) -> float:
    return float(number.replace(",", "")) * _UNIT_MULTIPLIERS.get(unit or "", 1)


def employee_bucket(raw: str | None) -> str | None:
    """Map headcount text such as '120' or '1,001-5,000 employees' to a fixed bucket."""
    text = (raw or "").lower()
    if not text or _PROFILE_COUNT_RE.search(text):
        return None

    if match := _COUNT_RANGE_RE.search(text):
        low_number, low_unit, high_number, high_unit = match.groups()
        high = _count_value(high_number, high_unit)
        count = (_count_value(low_number, low_unit or high_unit) + high) / 2
    elif match := _COUNT_RE.search(text):
        count = _count_value(match.group(1), match.group(2))
        if text[match.end() :].lstrip().startswith("+") or _ABOVE_RE.search(text):
            count += 1
    else:
        return None

    if count <= 0:
        return None
    return next((bucket for limit, bucket in EMPLOYEE_BUCKETS if count <= limit), "10000+")


def _without_ownership_text(summary: Sourced) -> Sourced:
    """Drop sentences about ownership. The ownership field carries that fact."""
    if summary.value is None:
        return summary
    sentences = re.split(r"(?<=[.!?])\s+", summary.value)
    kept = " ".join(s for s in sentences if not _OWNERSHIP_TEXT_RE.search(s))
    return summary.model_copy(update={"value": kept}) if kept else Sourced.empty()


def apply_field_rules(result: ExtractionResult, own_hosts: tuple[str, ...]) -> ExtractionResult:
    update: dict = {
        "website_summary": _without_ownership_text(result.website_summary),
        "location": _pick_locations(result.location, own_hosts),
        "recent_news": [item for item in result.recent_news if is_recent_news(item, date.today())],
    }
    if employee_bucket(result.employee_count.value) is None:
        update["employee_count"] = Sourced.empty()

    # "Own site" has to be true of the URL, not just the field the LLM put it in.
    own_industry = result.own_site_industry
    if own_industry.value is not None and not _host_in(_source_host(own_industry), own_hosts):
        update["own_site_industry"] = Sourced.empty()
        if result.third_party_industry.value is None:
            update["third_party_industry"] = own_industry
    return result.model_copy(update=update)


def _clear_funding_unless_private(result: ExtractionResult) -> ExtractionResult:
    if result.ownership.value in ("public", "acquired"):
        return result.model_copy(update={"funding": Sourced.empty()})
    return result


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
    own_hosts = _own_hosts(state)
    result = apply_field_rules(apply_confidence_rules(result, own_hosts), own_hosts)
    errors = [warning] if warning else []
    if dropped:
        errors.append(f"validate: dropped {dropped} fact(s) citing a URL that was not collected")
    return {
        "extraction": _clear_funding_unless_private(_merge_extractions(previous, result)),
        "cost_usd": cost,
        "errors": errors,
    }


def _resolved_industry(extraction: ExtractionResult) -> Sourced:
    """The company's own description wins over a third party's category."""
    own = extraction.own_site_industry
    return own if own.value else extraction.third_party_industry


async def validate(state: EnrichState) -> EnrichState:
    extraction = state.get("extraction")
    if extraction is None:
        return {"missing_fields": list(KEY_FIELDS)}
    values = {
        "industry": _resolved_industry(extraction).value,
        "employee_range": extraction.employee_count.value,
    }
    return {"missing_fields": [field for field in KEY_FIELDS if values[field] is None]}


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


def _ownership_fact(ownership: OwnershipFact) -> Sourced:
    value = ownership.value
    if value == "acquired":
        value = f"acquired_by: {ownership.acquired_by or 'unknown'}"
    return Sourced(value=value, source_url=ownership.source_url, confidence=ownership.confidence)


def _lead_fields(extraction: ExtractionResult | None) -> dict:
    """Turn what the LLM extracted into the fields the API returns."""
    if extraction is None:
        return {}
    count = extraction.employee_count
    return {
        **extraction.model_dump(
            exclude={"own_site_industry", "third_party_industry", "employee_count", "ownership"}
        ),
        "industry": _resolved_industry(extraction),
        "employee_range": Sourced(
            value=employee_bucket(count.value),
            source_url=count.source_url,
            confidence=count.confidence,
        ),
        "ownership": _ownership_fact(extraction.ownership),
    }


def _build_lead(state: EnrichState, is_free: bool, latency_ms: int) -> EnrichedLead:
    settings = get_settings()
    extraction = state.get("extraction")
    return EnrichedLead(
        **_lead_fields(extraction),
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
