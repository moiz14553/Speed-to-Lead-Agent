import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

DEBUG_DIR = Path("debug")


def _safe_name(text: str) -> str:
    return re.sub(r"[^\w.@-]", "_", text).strip(".") or "unknown"


def save_raw(run_id: str, state: dict) -> None:
    """Write everything one lead collected, including what the relevance filter dropped."""
    dropped = set(state.get("dropped_sources") or [])
    extraction = state.get("extraction")
    payload = {
        "email": state.get("email"),
        "company_hint": state.get("company_hint"),
        "domain": state.get("domain"),
        "target": state.get("target"),
        "site_name": state.get("site_name"),
        "errors": state.get("errors"),
        "website_lookup": [doc.model_dump() for doc in state.get("lookup_results") or []],
        "sources": [
            {**doc.model_dump(), "kept_by_relevance_filter": doc.url not in dropped}
            for doc in state.get("sources") or []
        ],
        "raw_html": state.get("raw_html") or {},
        "extraction": extraction.model_dump() if extraction else None,
    }
    try:
        folder = DEBUG_DIR / _safe_name(run_id)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{_safe_name(state.get('email') or 'unknown')}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        logger.exception("Could not save raw data for run %s", run_id)
