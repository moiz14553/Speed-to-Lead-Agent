from fastapi.testclient import TestClient

from app.main import app
from app.services.llm import LLMError, LLMWaitTooLongError

client = TestClient(app)


def post(**overrides):
    body = {"email": "jo@acme.com", "name": "Jo", "company_hint": None, **overrides}
    return client.post("/enrich", json=body)


def test_returns_full_lead(external):
    response = post()

    assert response.status_code == 200
    lead = response.json()
    assert lead["is_free_email"] is False
    assert lead["company_name"]["value"] == "Acme"
    assert lead["company_name"]["source_url"] == "https://acme.com/"
    assert lead["company_name"]["confidence"] == "high"
    assert lead["domain"]["value"] == "acme.com"
    assert [t["value"] for t in lead["tech_stack"]] == ["Shopify"]
    assert lead["recent_news"][0]["source_url"] == "https://news.example.org/acme"
    assert lead["latency_ms"] >= 0
    assert lead["errors"] == []


def test_reports_provider_and_model(external, settings):
    assert post().json()["llm_provider"] == "gemini"
    assert post().json()["llm_model"] == "gemini-3.8-flash"

    settings.llm_provider = "anthropic"
    lead = post().json()
    assert lead["llm_provider"] == "anthropic"
    assert lead["llm_model"] == "claude-haiku-5-5"


def test_cost_adds_llm_and_searches(external, settings):
    settings.tavily_cost_per_search_usd = 0.008
    # Three searches at 0.008 plus one LLM call at 0.01 from the fake.
    assert post().json()["cost_usd"] == 0.034


def test_run_id_header_names_the_debug_folder(external, settings, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    settings.debug_save_raw = True

    response = client.post(
        "/enrich", json={"email": "jo@acme.com", "name": "Jo"}, headers={"X-Run-Id": "run-42"}
    )

    assert response.status_code == 200
    assert (tmp_path / "debug" / "run-42" / "jo@acme.com.json").exists()


def test_invalid_email_returns_422(external):
    assert post(email="not-an-email").status_code == 422


def test_missing_name_returns_422(external):
    assert client.post("/enrich", json={"email": "jo@acme.com"}).status_code == 422


def test_long_provider_wait_returns_200_with_error(external):
    external.llm_results = [LLMWaitTooLongError("gave up, provider asked to wait 70919s")]

    response = post()

    assert response.status_code == 200
    assert "llm: gave up, provider asked to wait 70919s" in response.json()["errors"]


def test_llm_failure_returns_502_with_provider(external):
    external.llm_results = [LLMError("rate limited after 4 attempts")]

    response = post()

    assert response.status_code == 502
    detail = response.json()["detail"]
    assert "rate limited" in detail["message"]
    assert detail["llm_provider"] == "gemini"
    assert detail["llm_model"] == "gemini-3.8-flash"
