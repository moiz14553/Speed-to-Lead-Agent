import re
from typing import Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

Confidence = Literal["high", "medium", "low"]

_EMAIL_RE = re.compile(r"^[^@\s]+@([a-z0-9-]+\.)+[a-z]{2,}$", re.IGNORECASE)


class EnrichRequest(BaseModel):
    email: str
    name: str
    company_hint: str | None = None

    @field_validator("email")
    @classmethod
    def check_email(cls, value: str) -> str:
        value = value.strip()
        if not _EMAIL_RE.match(value):
            raise ValueError("not a valid email address")
        return value


class Sourced(BaseModel):
    """One fact with where it came from. All three are null when it was not found."""

    value: str | None = Field(description="The fact, or null if the sources do not state it.")
    source_url: str | None = Field(description="Exact URL of the source stating the fact.")
    confidence: Confidence | None = Field(
        description="high: stated on the company's own site. medium: stated by a credible "
        "third party. low: weak, indirect or outdated mention, or the sources disagree."
    )

    @model_validator(mode="after")
    def clear_when_empty(self) -> Self:
        if self.value is None:
            self.source_url = None
            self.confidence = None
        return self

    @classmethod
    def empty(cls) -> "Sourced":
        return cls(value=None, source_url=None, confidence=None)


class OwnershipFact(Sourced):
    value: Literal["private", "public", "acquired"] | None = Field(
        description="Ownership as stated in the sources, or null if they do not say."
    )
    acquired_by: str | None = Field(description="The acquiring company, if acquired and stated.")


class NewsItem(Sourced):
    published_date: str | None = Field(
        description="Date the source gives for the item, as YYYY-MM-DD or YYYY-MM. "
        "Null if the source gives no date."
    )


class ExtractionResult(BaseModel):
    """What the LLM fills in. Every field is required so null is an explicit answer."""

    company_name: Sourced
    website_summary: Sourced = Field(
        description="2-3 lines on what the company does. Nothing about who owns it."
    )
    own_site_industry: Sourced = Field(
        description="Short phrase for what the company sells, from its own site. Null if its "
        "own site does not say."
    )
    third_party_industry: Sourced = Field(description="Industry category from a third party.")
    employee_count: Sourced = Field(
        description="Headcount text exactly as stated, e.g. '120' or '1,001-5,000 employees'."
    )
    location: list[Sourced] = Field(description="Headquarters first, then up to 2 other offices.")
    ownership: OwnershipFact
    funding: Sourced = Field(description="Private funding rounds (seed, Series A, etc.), as stated.")
    recent_news: list[NewsItem] = Field(
        description="Up to 5 news items about the company itself from the last 12 months."
    )


class EnrichedLead(BaseModel):
    company_name: Sourced = Field(default_factory=Sourced.empty)
    domain: Sourced = Field(default_factory=Sourced.empty)
    website_summary: Sourced = Field(default_factory=Sourced.empty)
    industry: Sourced = Field(default_factory=Sourced.empty)
    employee_range: Sourced = Field(default_factory=Sourced.empty)
    location: list[Sourced] = []
    ownership: Sourced = Field(default_factory=Sourced.empty)
    tech_stack: list[Sourced] = []
    funding: Sourced = Field(default_factory=Sourced.empty)
    recent_news: list[NewsItem] = []
    is_free_email: bool
    cost_usd: float
    latency_ms: int
    llm_provider: str
    llm_model: str
    errors: list[str] = []


class SourceDoc(BaseModel):
    """Text collected from one URL, given to the LLM as evidence."""

    url: str
    title: str | None = None
    published_date: str | None = None
    text: str
