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
| LLM fails on the first extraction (bad key, refusal, retries used up) | 502 |
| LLM fails on the retry extraction | 200, keeps the first extraction, with an `llm:` entry in `errors` |
| Invalid email | 422 |

### Timeouts

`ENRICH_TIMEOUT_SECONDS` (default 30) limits the whole run. Each source also has its own limit:
`SCRAPE_TIMEOUT_SECONDS`, `TAVILY_TIMEOUT_SECONDS` and `LLM_TIMEOUT_SECONDS`.

If the budget runs out before the LLM has finished, the response has the domain and tech stack
but no LLM fields, because those fields only come from the LLM.

### Rate limits

LLM 429s and 5xx errors are retried with exponential backoff up to `LLM_MAX_RETRIES`, using the
delay the provider asks for when it gives one (capped at `LLM_BACKOFF_MAX_SECONDS`). Retries stop
when the time budget ends.

### Cost

`cost_usd` adds up LLM tokens (`LLM_RATES`), Tavily searches and Firecrawl pages. It leaves out
calls cut off by a timeout, because no token count comes back for them. A model missing from
`LLM_RATES` adds an entry to `errors` rather than silently counting as $0.

### Never inventing data

The LLM only sees the collected sources. Any fact citing a URL that was not collected is set to
null and counted in `errors`. Tech stack is detected from page HTML without the LLM. Each source
is capped at `SOURCE_CHAR_LIMIT` characters before it reaches the LLM.

## Tests

```bash
pytest
```

All external services are mocked, so no keys are needed. The tests run the real LangChain classes
for both providers and fake only the network call.
