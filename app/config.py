from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "anthropic": "claude-haiku-5-5"}


class ModelRate(BaseModel):
    input: float
    output: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    llm_provider: Literal["gemini", "anthropic"] = "gemini"
    llm_model: str | None = None
    google_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    llm_rates: dict[str, ModelRate] = {
        "gemini-3.8-flash": ModelRate(input=0, output=0),
        "claude-haiku-5-5": ModelRate(input=0.10, output=0.50),
    }

    tavily_api_key: SecretStr | None = None
    firecrawl_api_key: SecretStr | None = None
    tavily_cost_per_search_usd: float = 0.008
    firecrawl_cost_per_page_usd: float = 0.0

    enrich_timeout_seconds: float = 30
    scrape_timeout_seconds: float = 8
    tavily_timeout_seconds: float = 6
    llm_timeout_seconds: float = 15

    llm_max_retries: int = 3
    llm_backoff_base_seconds: float = 1
    llm_backoff_max_seconds: float = 20

    source_char_limit: int = 6000

    debug_save_raw: bool = False

    @property
    def resolved_model(self) -> str:
        return self.llm_model or DEFAULT_MODELS[self.llm_provider]


@lru_cache
def get_settings() -> Settings:
    return Settings()
