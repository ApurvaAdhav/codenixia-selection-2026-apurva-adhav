"""
app/data/store.py
-------------------
Tiny in-memory "current dataset" store, shared by the FastAPI app and the
Streamlit dashboard when they run in the same process (Streamlit calls the
agent/analytics functions directly). For the FastAPI server itself, this
module holds whatever dataset was last loaded via /upload, so /analyze and
/ask can use it without re-uploading every request.

Deliberately a plain module-level singleton (not a class you instantiate) -
there is exactly ONE active dataset per running process in this MVP, which
matches the challenge's "keep it simple" instruction. A production system
would key this by user/session/tenant.

NO SAMPLE/DEMO DATA: the store starts empty on every process start and stays
empty until a real file is uploaded via /upload (API) or the sidebar
uploader (Streamlit). has_dataset() is False and get_active_dataset()
returns None until then - callers must check before using the dataset.
"""
from __future__ import annotations

import pandas as pd

from app.data.processing import CleaningReport
from app.logging_config import setup_logging

logger = setup_logging(__name__)

_current_df: pd.DataFrame | None = None
_current_report: CleaningReport | None = None
_current_source_name: str | None = None


def set_active_dataset(df: pd.DataFrame, report: CleaningReport, source_name: str) -> None:
    global _current_df, _current_report, _current_source_name
    _current_df = df
    _current_report = report
    _current_source_name = source_name
    logger.info("Active dataset set from '%s' (%d rows).", source_name, len(df))


def get_active_dataset() -> pd.DataFrame | None:
    """Returns the active dataset, or None if nothing has been uploaded yet
    in this process. Callers must check has_dataset() / for None first."""
    return _current_df


def get_active_report() -> CleaningReport | None:
    return _current_report


def get_active_source_name() -> str | None:
    return _current_source_name


def has_dataset() -> bool:
    return _current_df is not None
