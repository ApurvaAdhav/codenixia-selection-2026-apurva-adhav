"""
app/config.py
--------------
Central place for all configuration. Everything is read from environment
variables (with sane defaults) so the project behaves the same in Docker,
locally, and during tests. Keeping this in one small file makes it easy to
explain in an interview: "all knobs live here."
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent          # project root

# Load .env explicitly from the project root (BASE_DIR/.env), rather than
# relying on load_dotenv()'s default cwd/stack-based auto-discovery.
#
# BUG THIS FIXES: the default `load_dotenv()` (no path argument) does NOT
# reliably find the .env file when the process is launched with a working
# directory other than the project root (e.g. `streamlit run
# /full/path/to/streamlit_app.py` from elsewhere, some IDE run configs, or
# certain process managers). When that happens, GEMINI_API_KEY silently
# reads as "" and the app falls back to the deterministic template with no
# error - which looks exactly like "the LLM integration doesn't work" even
# though the code and API key are both fine. Passing an explicit path
# removes that ambiguity entirely.
load_dotenv(dotenv_path=BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
LOG_DIR = BASE_DIR / "logs"

for _dir in (UPLOAD_DIR, FAISS_INDEX_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
# Provider can be "gemini" or "none" (offline/template fallback mode).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Guard against the extremely common setup mistake of copying .env.example
# to .env and forgetting to replace the placeholder value - without this,
# GEMINI_API_KEY would be a non-empty (but fake) string, the client would
# report itself as "enabled", and every call would fail in a way that's
# harder to diagnose than a clean "not configured" state.
_PLACEHOLDER_VALUES = {"", "your_gemini_api_key_here"}
if GEMINI_API_KEY in _PLACEHOLDER_VALUES:
    GEMINI_API_KEY = ""

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

# ---------------------------------------------------------------------------
# ML / Analysis configuration
# ---------------------------------------------------------------------------
ANOMALY_CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", "0.05"))  # expected % anomalies
FORECAST_PERIODS_DAYS = int(os.getenv("FORECAST_PERIODS_DAYS", "30"))

# ---------------------------------------------------------------------------
# RAG configuration
# ---------------------------------------------------------------------------
# NOTE: embeddings are TF-IDF (scikit-learn), fully local/offline - no model
# download required. See app/rag/retriever.py for the rationale.
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
