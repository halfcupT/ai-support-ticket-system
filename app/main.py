"""
REST API for the AI Support Ticket Intelligence system.

Run with:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health          - liveness + LLM reachability check
    POST /query            - ask a natural-language question about the tickets
    GET  /anomalies        - list detected anomalies (optionally filtered by type)
    GET  /stats             - quick aggregate stats (bonus, used by the UI dashboard)
    GET  /tickets           - paginated raw ticket listing (bonus, used by the UI table)
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app import anomaly, config
from app.data_store import DataStoreError, store
from app.llm_providers import get_llm
from app.query_engine import answer_question
from app.schemas import (
    AnomalyItem,
    AnomalyResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    StatsResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Support Ticket Intelligence API",
    description=(
        "Query a customer-support ticket dataset in natural language and "
        "surface anomalies (SLA breaches, abnormal resolution times)."
    ),
    version="1.0.0",
)

# Wide-open CORS since this is a local assessment prototype (Streamlit UI runs
# on a different port). Tighten this before ever deploying beyond localhost.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    try:
        df = store.load()
        logger.info("Startup: loaded %d tickets.", len(df))
    except DataStoreError as exc:
        # Don't crash the whole process on startup - /health will report the
        # problem clearly instead of the API failing to boot with a stack trace.
        logger.error("Startup data load failed: %s", exc)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        df = store.get()
        rows = len(df)
        data_ok = True
        detail = None
    except DataStoreError as exc:
        rows = 0
        data_ok = False
        detail = str(exc)

    llm = get_llm()
    llm_ok = llm.is_available()
    if not llm_ok and config.LLM_PROVIDER != "none":
        detail = (detail + " | " if detail else "") + (
            f"LLM provider '{config.LLM_PROVIDER}' is not reachable right now; "
            "/query will fall back to a templated answer with a warning."
        )

    status = "ok" if data_ok else "degraded"
    return HealthResponse(
        status=status,
        rows_loaded=rows,
        llm_provider=config.LLM_PROVIDER,
        llm_reachable=llm_ok,
        detail=detail,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        df = store.get()
    except DataStoreError as exc:
        raise HTTPException(status_code=500, detail=f"Data not available: {exc}") from exc

    llm = get_llm()
    result = answer_question(
        df=df,
        question=request.question,
        llm=llm,
        schema_description=store.schema_description(),
        phrase_with_llm=request.phrase_with_llm,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer_text,
        query_spec=result.spec,
        result=result.raw_result,
        used_llm_fallback=result.used_llm_fallback,
        warnings=result.warnings,
    )


@app.get("/anomalies", response_model=AnomalyResponse)
def anomalies(
    types: list[str] | None = Query(
        default=None,
        description=(
            "Restrict to specific anomaly types: long_resolution_time, "
            "slow_first_response, unresolved_sla_breach, "
            "low_satisfaction_despite_resolution. Omit for all."
        ),
    )
) -> AnomalyResponse:
    try:
        df = store.get()
    except DataStoreError as exc:
        raise HTTPException(status_code=500, detail=f"Data not available: {exc}") from exc

    type_set = set(types) if types else None
    found = anomaly.detect_all(df, types=type_set)
    return AnomalyResponse(
        summary=anomaly.summarize(found),
        anomalies=[AnomalyItem(**a.to_dict()) for a in found],
    )


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    try:
        df = store.get()
    except DataStoreError as exc:
        raise HTTPException(status_code=500, detail=f"Data not available: {exc}") from exc

    def safe_mean(col: str) -> float | None:
        series = df[col].dropna()
        return round(float(series.mean()), 3) if not series.empty else None

    return StatsResponse(
        total_tickets=len(df),
        by_status=df["status"].value_counts().to_dict(),
        by_priority=df["priority"].value_counts().to_dict(),
        by_category=df["category"].value_counts().to_dict(),
        avg_response_time_hrs=safe_mean("response_time_hrs"),
        avg_resolution_time_hrs=safe_mean("resolution_time_hrs"),
        avg_customer_rating=safe_mean("customer_rating"),
    )


@app.get("/tickets")
def tickets(
    status: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict:
    try:
        df = store.get()
    except DataStoreError as exc:
        raise HTTPException(status_code=500, detail=f"Data not available: {exc}") from exc

    filtered = df
    if status:
        filtered = filtered[filtered["status"] == status]
    if category:
        filtered = filtered[filtered["category"] == category]
    if priority:
        filtered = filtered[filtered["priority"] == priority]

    cols = [c for c in df.columns if c != "age_hrs"]
    page = filtered[cols].iloc[offset : offset + limit]
    return {
        "total_matching": int(len(filtered)),
        "offset": offset,
        "limit": limit,
        "tickets": page.to_dict(orient="records"),
    }
