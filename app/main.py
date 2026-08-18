"""
app/main.py
------------
FastAPI application entrypoint.

Run directly:
    uvicorn app.main:app --reload --port 8000

Or via Docker (see Dockerfile).
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.logging_config import setup_logging

logger = setup_logging(__name__)

app = FastAPI(
    title="AI Business Intelligence & Decision Support Assistant",
    description="CSV/Excel -> Pandas cleaning -> ML analysis -> AI Agent -> RAG -> LLM -> Structured insights",
    version="1.0.0",
)

# Permissive CORS since this is a local demo/interview project with no auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
def on_startup() -> None:
    """No dataset is bundled or auto-loaded - the app intentionally starts
    empty. Call POST /upload with a CSV/Excel file before /analyze or /ask
    will work (both return 400 with a clear message until then)."""
    logger.info("Startup complete: no active dataset yet - POST /upload to begin.")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all so an unexpected error returns clean JSON instead of a
    raw traceback / connection reset - required by 'error handling' in the
    challenge spec."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


@app.get("/")
def root():
    return {
        "service": "AI Business Intelligence & Decision Support Assistant",
        "docs": "/docs",
        "endpoints": ["/health", "/upload", "/analyze", "/ask"],
    }
