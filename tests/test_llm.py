"""Both providers run through their real LangChain classes. Only the network call is faked."""

import asyncio
import json
import time

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.exceptions import ModelAPIError, ModelInvalidRequestError, ModelRateLimitError
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from app.config import ModelRate, Settings
from app.models.enrichment import ExtractionResult, SourceDoc
from app.services import llm
from tests.conftest import full_extraction

PROVIDERS = {"gemini": ChatGoogleGenerativeAI, "anthropic": ChatAnthropic}
SOURCES = [SourceDoc(url="https://acme.com/", text="Acme makes widgets.")]
USAGE = {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}


def result_with(content: str, usage=USAGE, metadata=None) -> ChatResult:
    message = AIMessage(content=content, usage_metadata=usage, response_metadata=metadata or {})
    return ChatResult(generations=[ChatGeneration(message=message)])


def fake_network(monkeypatch, provider: str, responses: list):
    """Replace the provider's network call and return the list of calls it received."""
    calls = []
    queue = list(responses)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        calls.append({"messages": messages, "kwargs": kwargs})
        response = queue.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(PROVIDERS[provider], "_agenerate", _agenerate)
    return calls


def valid_response() -> ChatResult:
    return result_with(full_extraction().model_dump_json())


@pytest.mark.parametrize("provider,expected_class", PROVIDERS.items())
def test_builds_selected_provider_with_default_model(settings, provider, expected_class):
    settings.llm_provider = provider

    model = llm.build_chat_model(settings)

    assert isinstance(model, expected_class)
    assert settings.resolved_model == {
        "gemini": "gemini-3.8-flash",
        "anthropic": "claude-haiku-5-5",
    }[provider]


def test_gemini_is_the_default_provider():
    assert Settings().llm_provider == "gemini"


def test_llm_model_overrides_default(settings):
    settings.llm_model = "custom-model"
    assert settings.resolved_model == "custom-model"


@pytest.mark.parametrize("provider,key", [("gemini", "google"), ("anthropic", "anthropic")])
def test_missing_key_raises_clear_error(settings, provider, key):
    settings.llm_provider = provider
    setattr(settings, f"{key}_api_key", None)

    with pytest.raises(llm.LLMError, match="API_KEY is not set"):
        llm.build_chat_model(settings)


def test_client_retries_are_off(settings):
    # The Google client treats 1 as "no retries".
    assert llm.build_chat_model(settings).max_retries == 1
    settings.llm_provider = "anthropic"
    assert llm.build_chat_model(settings).max_retries == 0


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_structured_output_parses_for_both_providers(settings, monkeypatch, provider):
    settings.llm_provider = provider
    calls = fake_network(monkeypatch, provider, [valid_response()])

    result, _, warning = await llm.extract_facts("acme.com", SOURCES)

    assert isinstance(result, ExtractionResult)
    assert result.company_name.value == "Acme"
    assert result.recent_news[0].source_url == "https://news.example.org/acme"
    assert warning is None
    assert len(calls) == 1


async def test_gemini_requests_native_json_schema(settings, monkeypatch):
    calls = fake_network(monkeypatch, "gemini", [valid_response()])

    await llm.extract_facts("acme.com", SOURCES)

    kwargs = calls[0]["kwargs"]
    assert kwargs["response_mime_type"] == "application/json"
    schema = kwargs.get("response_json_schema") or kwargs.get("response_schema")
    assert "company_name" in json.dumps(schema)


async def test_anthropic_requests_json_schema_without_forced_tool(settings, monkeypatch):
    settings.llm_provider = "anthropic"
    calls = fake_network(monkeypatch, "anthropic", [valid_response()])

    await llm.extract_facts("acme.com", SOURCES)

    kwargs = calls[0]["kwargs"]
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert "tool_choice" not in kwargs
    assert "tools" not in kwargs


async def test_sources_reach_the_prompt_as_data(settings, monkeypatch):
    calls = fake_network(monkeypatch, "gemini", [valid_response()])

    await llm.extract_facts("acme.com", SOURCES)

    system, user = calls[0]["messages"]
    assert "untrusted" in system.content
    assert "https://acme.com/" in user.content
    assert "Acme makes widgets." in user.content


async def test_source_text_is_limited(settings, monkeypatch):
    settings.source_char_limit = 10
    calls = fake_network(monkeypatch, "gemini", [valid_response()])

    await llm.extract_facts("acme.com", [SourceDoc(url="https://a.com", text="x" * 50)])

    user = calls[0]["messages"][1].content
    assert "x" * 10 in user
    assert "x" * 11 not in user


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_retries_rate_limit_then_succeeds(settings, monkeypatch, provider):
    settings.llm_provider = provider
    responses = [ModelRateLimitError("429"), ModelRateLimitError("429"), valid_response()]
    calls = fake_network(monkeypatch, provider, responses)

    result, _, _ = await llm.extract_facts("acme.com", SOURCES)

    assert result.company_name.value == "Acme"
    assert len(calls) == 3


async def test_retries_provider_error(settings, monkeypatch):
    calls = fake_network(monkeypatch, "gemini", [ModelAPIError("503"), valid_response()])

    await llm.extract_facts("acme.com", SOURCES)

    assert len(calls) == 2


@pytest.mark.parametrize("provider", PROVIDERS)
async def test_gives_up_after_max_retries(settings, monkeypatch, provider):
    settings.llm_provider = provider
    settings.llm_max_retries = 2
    calls = fake_network(monkeypatch, provider, [ModelRateLimitError("429")] * 3)

    with pytest.raises(llm.LLMError, match="rate limited after 3 attempts"):
        await llm.extract_facts("acme.com", SOURCES)

    assert len(calls) == 3


async def test_does_not_retry_other_errors(settings, monkeypatch):
    calls = fake_network(monkeypatch, "gemini", [ModelInvalidRequestError("bad request")])

    with pytest.raises(llm.LLMError, match="ModelInvalidRequestError"):
        await llm.extract_facts("acme.com", SOURCES)

    assert len(calls) == 1


async def test_backoff_waits_grow_and_respect_the_cap(settings, monkeypatch):
    settings.llm_backoff_base_seconds = 2
    settings.llm_backoff_max_seconds = 5
    fake_network(monkeypatch, "gemini", [ModelRateLimitError("429")] * 4)
    waits = []

    async def sleep(seconds):
        waits.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", sleep)

    with pytest.raises(llm.LLMError):
        await llm.extract_facts("acme.com", SOURCES)

    assert 2 <= waits[0] <= 3
    assert 4 <= waits[1] <= 5
    assert waits[2] == 5


def test_backoff_uses_delay_from_gemini_error(settings):
    settings.llm_backoff_max_seconds = 60
    exc = ModelRateLimitError("429 RESOURCE_EXHAUSTED. Please retry in 23.5s.")
    assert llm._backoff_seconds(0, exc, settings) == 23.5


def test_backoff_uses_retry_delay_field(settings):
    settings.llm_backoff_max_seconds = 60
    exc = ModelRateLimitError("details: {'retryDelay': '41s'}")
    assert llm._backoff_seconds(0, exc, settings) == 41


def test_backoff_reads_hours_and_minutes_from_gemini_error(settings):
    exc = ModelRateLimitError("Quota exceeded. Please retry in 19h41m59.5s.")
    assert llm._retry_after_seconds(exc) == 19 * 3600 + 41 * 60 + 59.5


def test_backoff_cap_does_not_shorten_the_providers_wait(settings):
    settings.llm_backoff_max_seconds = 5
    exc = ModelRateLimitError("Please retry in 12s.")
    assert llm._backoff_seconds(0, exc, settings) == 12


async def test_gives_up_at_once_when_wait_is_longer_than_budget(settings, monkeypatch):
    calls = fake_network(
        monkeypatch, "gemini", [ModelRateLimitError("Please retry in 19h41m59.5s.")]
    )
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", sleep)

    with pytest.raises(llm.LLMWaitTooLongError, match=r"asked to wait 70920s with \d+s left"):
        await llm.extract_facts("acme.com", SOURCES, deadline=time.monotonic() + 30)

    assert len(calls) == 1
    assert sleeps == []


async def test_waits_when_providers_wait_fits_the_budget(settings, monkeypatch):
    calls = fake_network(
        monkeypatch, "gemini", [ModelRateLimitError("Please retry in 2s."), valid_response()]
    )
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(llm.asyncio, "sleep", sleep)

    await llm.extract_facts("acme.com", SOURCES, deadline=time.monotonic() + 30)

    assert len(calls) == 2
    assert sleeps == [2]


async def test_call_timeout_raises_timeout_error(settings, monkeypatch):
    settings.llm_timeout_seconds = 0.01

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        await asyncio.sleep(5)

    monkeypatch.setattr(ChatGoogleGenerativeAI, "_agenerate", _agenerate)

    with pytest.raises(llm.LLMTimeoutError):
        await llm.extract_facts("acme.com", SOURCES)


@pytest.mark.parametrize(
    "provider,metadata",
    [("gemini", {"finish_reason": "SAFETY"}), ("anthropic", {"stop_reason": "refusal"})],
)
async def test_unparseable_output_raises_with_stop_reason(settings, monkeypatch, provider, metadata):
    settings.llm_provider = provider
    fake_network(monkeypatch, provider, [result_with("not json", metadata=metadata)])

    with pytest.raises(llm.LLMError, match=list(metadata.values())[0]):
        await llm.extract_facts("acme.com", SOURCES)


async def test_gemini_free_tier_costs_nothing(settings, monkeypatch):
    fake_network(monkeypatch, "gemini", [valid_response()])

    _, cost, warning = await llm.extract_facts("acme.com", SOURCES)

    assert cost == 0
    assert warning is None


async def test_anthropic_cost_from_tokens_and_rates(settings, monkeypatch):
    settings.llm_provider = "anthropic"
    fake_network(monkeypatch, "anthropic", [valid_response()])

    _, cost, _ = await llm.extract_facts("acme.com", SOURCES)

    # 1000 input at $0.10/M plus 100 output at $0.50/M.
    assert cost == pytest.approx(0.00015)


async def test_paid_gemini_rate_from_settings(settings, monkeypatch):
    settings.llm_rates["gemini-3.8-flash"] = ModelRate(input=1, output=2)
    fake_network(monkeypatch, "gemini", [valid_response()])

    _, cost, _ = await llm.extract_facts("acme.com", SOURCES)

    assert cost == pytest.approx(0.0012)


async def test_model_without_rate_warns(settings, monkeypatch):
    settings.llm_model = "unknown-model"
    fake_network(monkeypatch, "gemini", [valid_response()])

    _, cost, warning = await llm.extract_facts("acme.com", SOURCES)

    assert cost == 0
    assert "no rate for unknown-model" in warning


def test_rates_load_from_env(monkeypatch):
    monkeypatch.setenv("LLM_RATES", '{"some-model":{"input":3,"output":4}}')
    assert Settings().llm_rates["some-model"].output == 4


def test_empty_env_values_are_ignored(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "")
    assert Settings().resolved_model == "gemini-3.8-flash"


def test_key_is_not_exposed_in_repr(settings):
    settings.google_api_key = SecretStr("very-secret")
    assert "very-secret" not in repr(settings)
