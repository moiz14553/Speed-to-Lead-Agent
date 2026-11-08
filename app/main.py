from fastapi import FastAPI, Header, HTTPException

from app.agents.enrichment import run_enrichment
from app.config import get_settings
from app.models.enrichment import EnrichedLead, EnrichRequest
from app.services.llm import LLMError

app = FastAPI(title="Speed-to-Lead Agent")


@app.post("/enrich", response_model=EnrichedLead)
async def enrich(
    request: EnrichRequest, x_run_id: str | None = Header(default=None)
) -> EnrichedLead:
    try:
        return await run_enrichment(request, x_run_id)
    except LLMError as exc:
        settings = get_settings()
        raise HTTPException(
            status_code=502,
            detail={
                "message": f"LLM failed: {exc}",
                "llm_provider": settings.llm_provider,
                "llm_model": settings.resolved_model,
            },
        ) from exc
