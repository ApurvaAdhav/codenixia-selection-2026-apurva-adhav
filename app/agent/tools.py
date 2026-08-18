"""
app/agent/tools.py
-------------------
Defines the six tools required by the challenge:
  analyze_sales(), analyze_region(), analyze_products(),
  detect_anomalies(), predict_next_month(), search_business_knowledge()

Each tool is a small, pure function: (DataFrame, **params) -> dict/list.
This file is intentionally "boring" - it just wires app/data/* and
app/rag/* functions into a single TOOLS registry with JSON-schema-like
descriptions, so the agent (app/agent/core.py) can:
  a) decide which tool(s) to call based on the user's question, and
  b) call them dynamically by name.

Keeping tool logic OUT of the agent's decision logic is what makes this
"simple, modular, and easy to modify" - you can add a 7th tool here without
touching the agent's control flow.
"""
from __future__ import annotations

import pandas as pd

from app.data.analytics import analyze_sales as _analyze_sales
from app.data.analytics import analyze_region as _analyze_region
from app.data.analytics import analyze_products as _analyze_products
from app.data.anomaly import detect_anomalies as _detect_anomalies
from app.data.forecast import predict_next_period as _predict_next_period
from app.rag.retriever import search_business_knowledge as _search_business_knowledge
from app.logging_config import setup_logging

logger = setup_logging(__name__)


def analyze_sales(df: pd.DataFrame, period_days: int = 30) -> dict:
    """Overall revenue/profit/quantity trend vs the prior period, with top
    region and product contributors to the change."""
    return _analyze_sales(df, period_days=period_days)


def analyze_region(df: pd.DataFrame, region: str, period_days: int = 30) -> dict:
    """Deep-dive on one region: revenue trend + which products/segments
    drove the change in that region."""
    return _analyze_region(df, region=region, period_days=period_days)


def analyze_products(df: pd.DataFrame, product: str | None = None, period_days: int = 30) -> dict:
    """Deep-dive on one product (across regions), or if no product is given,
    ranks all products by revenue change."""
    return _analyze_products(df, product=product, period_days=period_days)


def detect_anomalies(df: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """Runs Isolation Forest per Region+Product daily revenue series and
    returns the most anomalous data points."""
    anomalies = _detect_anomalies(df)
    return anomalies[:top_n]


def predict_next_month(df: pd.DataFrame, region: str | None = None, product: str | None = None) -> dict:
    """Forecasts revenue for the next ~30 days using a linear trend model,
    optionally scoped to one region and/or product."""
    return _predict_next_period(df, periods=30, region=region, product=product)


def search_business_knowledge(query: str, top_k: int = 3) -> list[dict]:
    """Searches the local FAISS knowledge base (policies, KPI definitions,
    past incident reports) for context relevant to `query`."""
    return _search_business_knowledge(query, top_k=top_k)


# Registry used by the agent to look tools up by name + know what each does.
# NOTE: df-consuming tools are called with the current dataset already bound
# by the agent (see agent/core.py:_call_tool), so `description` here is
# written from the agent's point of view (what question does this answer).
TOOL_REGISTRY = {
    "analyze_sales": {
        "fn": analyze_sales,
        "needs_df": True,
        "description": "Overall sales health: revenue/profit/quantity change vs prior period, plus top region/product contributors. Use for general 'why did sales change' questions.",
        "params": ["period_days"],
    },
    "analyze_region": {
        "fn": analyze_region,
        "needs_df": True,
        "description": "Deep-dive on a specific region. Use when the user names a region (e.g. 'What about Region West?').",
        "params": ["region", "period_days"],
    },
    "analyze_products": {
        "fn": analyze_products,
        "needs_df": True,
        "description": "Deep-dive on a specific product, or rank all products by change if no product given. Use when the user names a product or asks 'which product'.",
        "params": ["product", "period_days"],
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "needs_df": True,
        "description": "Finds statistically unusual daily revenue points per region/product using Isolation Forest. Use for 'anomaly' or 'unusual' questions, or to explain a drop.",
        "params": ["top_n"],
    },
    "predict_next_month": {
        "fn": predict_next_month,
        "needs_df": True,
        "description": "Forecasts next ~30 days of revenue via linear trend. Use for 'what's expected next month' questions.",
        "params": ["region", "product"],
    },
    "search_business_knowledge": {
        "fn": search_business_knowledge,
        "needs_df": False,
        "description": "Searches company policies, KPI definitions, and past incident reports. Use to explain WHY something might have happened, or what policy applies.",
        "params": ["query", "top_k"],
    },
}
