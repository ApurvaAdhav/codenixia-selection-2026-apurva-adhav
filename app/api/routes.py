"""
app/api/routes.py
-------------------
FastAPI routes required by the challenge:
    /health   - service + dataset + LLM status
    /upload   - upload a CSV/Excel sales file, clean it, make it the active dataset
    /analyze  - run the full ML analysis (KPIs, trend, anomalies, forecast) on the active dataset
    /ask      - natural-language question -> AI Agent -> structured answer

All routes are thin: they validate input, call into app/data/*, app/agent/*,
and return. No business logic lives here - that keeps routes.py easy to
read end-to-end during an interview walkthrough.
"""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.agent import core as agent_core
from app.agent.llm_client import get_llm_client
from app.api.schemas import (
    AnalyzeRequest, AnalyzeResponse, AskRequest, AskResponse,
    HealthResponse, UploadResponse,
)
from app.data import store
from app.data.analytics import analyze_sales, get_kpi_summary
from app.data.anomaly import detect_anomalies
from app.data.forecast import predict_next_period
from app.data.processing import DataValidationError, load_and_clean
from app.logging_config import setup_logging

logger = setup_logging(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + readiness check: confirms the service is up, whether a
    dataset is loaded, and whether the LLM is configured (vs. running in
    fallback-only mode)."""
    dataset_loaded = store.has_dataset()
    df = store.get_active_dataset() if dataset_loaded else None
    llm = get_llm_client()

    return HealthResponse(
        status="ok",
        dataset_loaded=dataset_loaded,
        dataset_rows=len(df) if df is not None else 0,
        dataset_source=store.get_active_source_name(),
        llm_enabled=llm.enabled,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)) -> UploadResponse:
    """Upload a CSV/Excel sales file. It is cleaned/validated and becomes
    the ACTIVE dataset used by all subsequent /analyze and /ask calls."""
    try:
        file_bytes = await file.read()
        df, report = load_and_clean(file_bytes, file.filename)
    except DataValidationError as exc:
        logger.warning("Upload validation failed for '%s': %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error processing upload '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Internal error processing file: {exc}") from exc

    store.set_active_dataset(df, report, file.filename)
    # New dataset -> old conversation context no longer applies.
    agent_core.reset_session("default")

    return UploadResponse(
        filename=file.filename,
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        dropped_missing_required=report.dropped_missing_required,
        dropped_duplicates=report.dropped_duplicates,
        profit_recomputed=report.profit_recomputed,
        notes=report.notes,
        message=f"Dataset loaded and cleaned successfully: {report.rows_out} usable rows.",
    )


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """Runs the full ML analysis pipeline on the ACTIVE dataset: KPI
    summary, period-over-period sales analysis, anomaly detection, and a
    next-period forecast. This is the 'ML Analysis' step surfaced directly
    (independent of the conversational agent) for the dashboard's charts."""
    try:
        df = store.get_active_dataset()
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No dataset loaded. Upload a CSV/Excel file via /upload first.")

        kpi_summary = get_kpi_summary(df)
        sales_analysis = analyze_sales(df, period_days=request.period_days)
        anomalies = detect_anomalies(df)[:10]
        forecast = predict_next_period(df, periods=30)

        return AnalyzeResponse(
            kpi_summary=kpi_summary,
            sales_analysis=sales_analysis,
            anomalies=anomalies,
            forecast=forecast,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error during /analyze")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Natural-language question -> AI Agent (tool routing + RAG + LLM) ->
    structured, grounded answer. Multi-turn context is kept per session_id."""
    try:
        df = store.get_active_dataset()
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No dataset loaded. Upload one first via /upload.")

        result = agent_core.ask(question=request.question, df=df, session_id=request.session_id)

        return AskResponse(
            answer=result["answer"],
            tool_calls=result["tool_calls"],
            tool_results=result["tool_results"],
            entities=result["entities"],
            used_llm=result["used_llm"],
            session_id=request.session_id,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error during /ask for question=%r", request.question)
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {exc}") from exc
