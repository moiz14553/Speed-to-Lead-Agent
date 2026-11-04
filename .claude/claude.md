# Speed-to-Lead Agent

## What this is
An AI system that responds to inbound leads within 60 seconds on WhatsApp.
Flow: form submitted → enrich company data → WhatsApp first message → AI conversation
qualifies the lead (need, budget, timeline) → score Hot/Warm/Cold → book meeting / alert team → update CRM.

## Stack
- Python 3.12, FastAPI, LangGraph, Pydantic v2
- LLM: configurable with LLM_PROVIDER: gemini (langchain-google-genai, default) or anthropic (langchain-anthropic); model name from env var
- Search: Tavily; website scraping: Firecrawl (or httpx + BeautifulSoup fallback)
- DB: Postgres (Supabase) via SQLAlchemy; LangGraph Postgres checkpointer for chat state
- WhatsApp: Twilio (sandbox for dev)
- Orchestration: n8n (calls FastAPI endpoints via HTTP)
- CRM: HubSpot free; alerts: Slack incoming webhook

## Rules
- All secrets from .env (provide .env.example). Never hardcode keys.
- Every LLM output that feeds code must use structured output with Pydantic models.
- Every enriched fact must store its source URL and a confidence (high/medium/low).
- Never invent data: if not found, return null.
- Write pytest tests for every endpoint. Mock external APIs in tests.
- Keep it simple: no extra abstractions or frameworks I didn't ask for.
- Structure: app/main.py, app/agents/, app/services/, app/models/, tests/