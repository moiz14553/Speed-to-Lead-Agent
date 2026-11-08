# Speed-to-Lead Agent

Phase 1: the enrichment agent. `POST /enrich` takes a lead's email and returns what can be
found about their company. Every fact includes its source URL and a confidence level.

```
resolve_domain → scrape_site + search_company + search_funding_news (parallel)
               → extract → validate → (key fields missing, once) targeted_search → extract
```

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Fill in `.env`. You need `TAVILY_API_KEY` and the key for your LLM provider.
`FIRECRAWL_API_KEY` is optional.

| `LLM_PROVIDER` | Key | Default model |
|---|---|---|
| `gemini` (default) | `GOOGLE_API_KEY` | `gemini-3.8-flash` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |

Set `LLM_MODEL` to use a different model. Free-tier quotas vary by Google project and change
often, so check yours in Google AI Studio.

## Run

```bash
uvicorn app.main:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/enrich -H "Content-Type: application/json" \
  -d '{"email": "jo@stripe.com", "name": "Jo", "company_hint": null}'
```

`/enrich` has no authentication yet, so keep it local.

## Check accuracy

Edit `scripts/test_emails.csv`, start the server, then run:

```bash
python scripts/test_enrich.py
```

Each run adds its rows to `results.csv`, with a `run_id`, `llm_provider` and `llm_model` on
every row. To compare providers, change `LLM_PROVIDER` in `.env`, restart the server and run the
script again. Filter or sort `results.csv` by `email` to see the runs side by side.

## How the API behaves

| Situation | Response |
|---|---|
| Free email, no `company_hint` | 200, `is_free_email=true`, empty fields, no external calls |
| A source fails or times out | 200, the run continues, the failure is listed in `errors` |
| Time budget runs out | 200, with what was found so far and a `timeout:` entry in `errors` |
| LLM call times out | 200, without LLM fields, with an `llm:` entry in `errors` |
| Provider asks for a longer wait than the budget has left | 200, without LLM fields, with an `llm:` entry in `errors`; no more LLM calls |
| LLM fails on the first extraction (bad key, refusal, retries used up) | 502 |
| LLM fails on the retry extraction | 200, keeps the first extraction, with an `llm:` entry in `errors` |
| Invalid email | 422 |

### Scraping

A site answering 429 gets one short retry, unless it asks for a wait longer than 3 seconds.

### Timeouts

`ENRICH_TIMEOUT_SECONDS` (default 30) limits the whole run. Each source also has its own limit:
`SCRAPE_TIMEOUT_SECONDS`, `TAVILY_TIMEOUT_SECONDS` and `LLM_TIMEOUT_SECONDS`.

If the budget runs out before the LLM has finished, the response has the domain and tech stack
but no LLM fields, because those fields only come from the LLM.

### Rate limits

LLM 429s and 5xx errors are retried with exponential backoff up to `LLM_MAX_RETRIES`, using the
delay the provider asks for when it gives one. `LLM_BACKOFF_MAX_SECONDS` caps only our own
backoff, not the provider's delay. If that delay is longer than what is left of the time budget,
such as a used-up daily quota, the run gives up at once rather than retrying.

### Cost

`cost_usd` adds up LLM tokens (`LLM_RATES`), Tavily searches and Firecrawl pages. It leaves out
calls cut off by a timeout, because no token count comes back for them. A model missing from
`LLM_RATES` adds an entry to `errors` rather than silently counting as $0.

### Never inventing data

The LLM only sees the collected sources. Any fact citing a URL that was not collected is set to
null and counted in `errors`. Tech stack is detected from page HTML without the LLM. Each source
is capped at `SOURCE_CHAR_LIMIT` characters before it reaches the LLM.

### Field rules

| Field | Rule |
|---|---|
| any `confidence` | `high` only from the company's own domain. A short allowlist of trusted third parties (LinkedIn, Wikipedia, Crunchbase, major news sites) reaches `medium`. Every other site is `low`. If sources disagree on a field, `low` |
| `domain` from a company name | Picks the search result whose domain best matches the name, preferring an exact `.com`. Null if none matches. Search-provider pages are never used |
| `industry` | The company's own description of what it sells, before any third-party label |
| `location` | A list, headquarters first, at most 3 entries. The company's own site is preferred, and annual-report branch lists are ignored |
| `employee_range` | The LLM extracts the raw headcount text and code maps it to `1-10`, `11-50`, `51-200`, `201-500`, `501-1000`, `1001-5000`, `5001-10000` or `10000+`. Profile counts don't qualify |
| `ownership` | `private`, `public` or `acquired_by: X`, with its own source. Not repeated in `website_summary` |
| `funding` | Private rounds only. Code sets it to null when `ownership` is public or acquired |
| `recent_news` | About the company itself, with a date from the source, within the last 12 months |

## Tests

```bash
pytest
```

All external services are mocked, so no keys are needed. The tests run the real LangChain classes
for both providers and fake only the network call.
