"""
app/api/schemas.py
--------------------
Pydantic models for request/response validation. Kept separate from
routes.py so the API contract is easy to scan in one place - useful when
explaining the API surface in an interview.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    dataset_loaded: bool
    dataset_rows: int
    dataset_source: Optional[str] = None
    llm_enabled: bool


class UploadResponse(BaseModel):
    filename: str
    rows_in: int
    rows_out: int
    dropped_missing_required: int
    dropped_duplicates: int
    profit_recomputed: int
    notes: list[str]
    message: str


class AnalyzeRequest(BaseModel):
    period_days: int = Field(default=30, ge=7, le=365)


class AnalyzeResponse(BaseModel):
    kpi_summary: dict[str, Any]
    sales_analysis: dict[str, Any]
    anomalies: list[dict[str, Any]]
    forecast: dict[str, Any]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: str = Field(default="default")


class AskResponse(BaseModel):
    answer: str
    tool_calls: list[str]
    tool_results: dict[str, Any]
    entities: dict[str, Any]
    used_llm: bool
    session_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
