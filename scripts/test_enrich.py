"""Run /enrich on a CSV of emails and add the results to results.csv for manual checking.

The server must be running. Rows are appended with a run_id and the provider and model,
so runs with different LLM settings can be compared side by side in one file.

Usage: python scripts/test_enrich.py [--input FILE] [--output FILE] [--url URL]
"""

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

SINGLE_FIELDS = [
    "company_name",
    "domain",
    "website_summary",
    "industry",
    "employee_range",
    "location",
    "funding",
]
LIST_FIELDS = ["tech_stack", "recent_news"]
COLUMNS = [
    "run_id",
    "llm_provider",
    "llm_model",
    "email",
    "name",
    "company_hint",
    "http_status",
    "is_free_email",
    *[f"{field}{suffix}" for field in SINGLE_FIELDS for suffix in ("", "_source", "_confidence")],
    *[f"{field}{suffix}" for field in LIST_FIELDS for suffix in ("", "_sources", "_confidences")],
    "recent_news_dates",
    "cost_usd",
    "latency_ms",
    "errors",
]
# Longer than the server's time budget, so the server decides when a run stops.
REQUEST_TIMEOUT_SECONDS = 180


def to_columns(lead: dict) -> dict:
    row = {
        key: lead.get(key)
        for key in ("llm_provider", "llm_model", "is_free_email", "cost_usd", "latency_ms")
    }
    for field in SINGLE_FIELDS:
        fact = lead[field]
        row[field] = fact["value"]
        row[f"{field}_source"] = fact["source_url"]
        row[f"{field}_confidence"] = fact["confidence"]
    for field in LIST_FIELDS:
        row[field] = " | ".join(fact["value"] for fact in lead[field])
        row[f"{field}_sources"] = " | ".join(fact["source_url"] for fact in lead[field])
        row[f"{field}_confidences"] = " | ".join(fact["confidence"] for fact in lead[field])
    row["recent_news_dates"] = " | ".join(fact["published_date"] for fact in lead["recent_news"])
    row["errors"] = " | ".join(lead["errors"])
    return row


def enrich_row(client: httpx.Client, url: str, lead: dict) -> dict:
    payload = {
        "email": lead["email"],
        "name": lead["name"],
        "company_hint": lead.get("company_hint") or None,
    }
    try:
        response = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        return {"errors": f"request failed: {exc!r}"}

    row = {"http_status": response.status_code}
    if response.status_code == 200:
        return {**row, **to_columns(response.json())}

    detail = response.json().get("detail")
    if isinstance(detail, dict):
        row["llm_provider"] = detail.get("llm_provider")
        row["llm_model"] = detail.get("llm_model")
        detail = detail.get("message")
    return {**row, "errors": str(detail)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("scripts/test_emails.csv"))
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--url", default="http://127.0.0.1:8000/enrich")
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as file:
        leads = list(csv.DictReader(file))

    is_new_file = not args.output.exists() or args.output.stat().st_size == 0
    if not is_new_file:
        with args.output.open(newline="", encoding="utf-8") as file:
            if next(csv.reader(file), None) != COLUMNS:
                sys.exit(f"{args.output} has different columns. Use another --output.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    with (
        args.output.open("a", newline="", encoding="utf-8") as file,
        httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client,
    ):
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if is_new_file:
            writer.writeheader()
        for index, lead in enumerate(leads, start=1):
            print(f"[{index}/{len(leads)}] {lead['email']}")
            row = enrich_row(client, args.url, lead)
            writer.writerow(
                {
                    "run_id": run_id,
                    "email": lead["email"],
                    "name": lead["name"],
                    "company_hint": lead.get("company_hint"),
                    **row,
                }
            )
            file.flush()
    print(f"Run {run_id}: {len(leads)} rows added to {args.output}")


if __name__ == "__main__":
    main()
