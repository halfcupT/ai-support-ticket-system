from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["How many tickets are currently open?"])
    phrase_with_llm: bool = Field(
        True, description="If false, skip the LLM phrasing step and return a templated answer."
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    query_spec: dict[str, Any]
    result: Any
    used_llm_fallback: bool
    warnings: list[str] = []


class AnomalyItem(BaseModel):
    ticket_id: str
    type: str
    detail: str
    severity: str
    data: dict[str, Any] = {}


class AnomalyResponse(BaseModel):
    summary: dict[str, Any]
    anomalies: list[AnomalyItem]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    rows_loaded: int
    llm_provider: str
    llm_reachable: bool
    detail: str | None = None


class StatsResponse(BaseModel):
    total_tickets: int
    by_status: dict[str, int]
    by_priority: dict[str, int]
    by_category: dict[str, int]
    avg_response_time_hrs: float | None
    avg_resolution_time_hrs: float | None
    avg_customer_rating: float | None
