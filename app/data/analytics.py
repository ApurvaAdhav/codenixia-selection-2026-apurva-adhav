"""
app/data/analytics.py
----------------------
The analytical "brains" behind the agent tools analyze_sales(), analyze_region(),
and analyze_products(). Everything here is plain pandas - no ML, no LLM. This
is the layer that guarantees the LLM "never invents numbers": every figure the
LLM is allowed to talk about must come from a function in this file (or
anomaly.py / forecast.py).

Core idea: compare the LAST period vs the PREVIOUS period of equal length
(e.g. last 30 days vs the 30 days before that) to answer "what changed".
"""
from __future__ import annotations

import pandas as pd

from app.logging_config import setup_logging

logger = setup_logging(__name__)


def _split_periods(df: pd.DataFrame, period_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (current_period, previous_period) of equal length,
    based on the max date present in the data."""
    max_date = df["Date"].max()
    current_start = max_date - pd.Timedelta(days=period_days - 1)
    previous_start = current_start - pd.Timedelta(days=period_days)
    previous_end = current_start - pd.Timedelta(days=1)

    current = df[(df["Date"] >= current_start) & (df["Date"] <= max_date)]
    previous = df[(df["Date"] >= previous_start) & (df["Date"] <= previous_end)]
    return current, previous


def _pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0 if current == 0 else 100.0
    return round((current - previous) / previous * 100, 1)


def analyze_sales(df: pd.DataFrame, period_days: int = 30) -> dict:
    """
    Overall sales health check: compares the most recent `period_days`
    window against the prior window of equal length.
    Powers: analyze_sales() tool + the top-level "Why did sales drop?" answer.
    """
    current, previous = _split_periods(df, period_days)

    cur_revenue = float(current["Revenue"].sum())
    prev_revenue = float(previous["Revenue"].sum())
    cur_profit = float(current["Profit"].sum())
    prev_profit = float(previous["Profit"].sum())
    cur_qty = float(current["Quantity"].sum())
    prev_qty = float(previous["Quantity"].sum())

    revenue_change_pct = _pct_change(cur_revenue, prev_revenue)
    profit_change_pct = _pct_change(cur_profit, prev_profit)
    qty_change_pct = _pct_change(cur_qty, prev_qty)

    # --- Top contributors to the change: by Region and by Product --------
    region_contrib = _contributor_breakdown(current, previous, "Region")
    product_contrib = _contributor_breakdown(current, previous, "Product")

    result = {
        "period_days": period_days,
        "current_period": {
            "start": current["Date"].min().strftime("%Y-%m-%d") if not current.empty else None,
            "end": current["Date"].max().strftime("%Y-%m-%d") if not current.empty else None,
            "revenue": round(cur_revenue, 2),
            "profit": round(cur_profit, 2),
            "quantity": round(cur_qty, 2),
        },
        "previous_period": {
            "start": previous["Date"].min().strftime("%Y-%m-%d") if not previous.empty else None,
            "end": previous["Date"].max().strftime("%Y-%m-%d") if not previous.empty else None,
            "revenue": round(prev_revenue, 2),
            "profit": round(prev_profit, 2),
            "quantity": round(prev_qty, 2),
        },
        "revenue_change_pct": revenue_change_pct,
        "profit_change_pct": profit_change_pct,
        "quantity_change_pct": qty_change_pct,
        "top_region_contributors": region_contrib[:3],
        "top_product_contributors": product_contrib[:3],
    }
    logger.info("analyze_sales: revenue_change_pct=%s", revenue_change_pct)
    return result


def _contributor_breakdown(current: pd.DataFrame, previous: pd.DataFrame, dim: str) -> list[dict]:
    """For a dimension (Region or Product), compute revenue change per value
    and sort by the ones that dropped/contributed the most (most negative first)."""
    cur_agg = current.groupby(dim)["Revenue"].sum()
    prev_agg = previous.groupby(dim)["Revenue"].sum()

    all_keys = set(cur_agg.index) | set(prev_agg.index)
    rows = []
    for key in all_keys:
        cur_val = float(cur_agg.get(key, 0))
        prev_val = float(prev_agg.get(key, 0))
        change_pct = _pct_change(cur_val, prev_val)
        rows.append({
            dim.lower(): key,
            "current_revenue": round(cur_val, 2),
            "previous_revenue": round(prev_val, 2),
            "change_pct": change_pct,
            "absolute_change": round(cur_val - prev_val, 2),
        })
    # Sort by absolute_change ascending -> biggest drops first
    rows.sort(key=lambda r: r["absolute_change"])
    return rows


def analyze_region(df: pd.DataFrame, region: str, period_days: int = 30) -> dict:
    """Deep-dive on a single region. Powers analyze_region() tool."""
    region_df = df[df["Region"].str.lower() == region.lower()]
    if region_df.empty:
        return {"error": f"No data found for region '{region}'.", "available_regions": sorted(df["Region"].unique().tolist())}

    current, previous = _split_periods(region_df, period_days)
    cur_revenue = float(current["Revenue"].sum())
    prev_revenue = float(previous["Revenue"].sum())

    product_contrib = _contributor_breakdown(current, previous, "Product")
    segment_contrib = _contributor_breakdown(current, previous, "Customer_Segment")

    return {
        "region": region,
        "period_days": period_days,
        "current_revenue": round(cur_revenue, 2),
        "previous_revenue": round(prev_revenue, 2),
        "revenue_change_pct": _pct_change(cur_revenue, prev_revenue),
        "top_product_contributors": product_contrib[:3],
        "top_segment_contributors": segment_contrib[:3],
    }


def analyze_products(df: pd.DataFrame, product: str | None = None, period_days: int = 30) -> dict:
    """
    If `product` is given, deep-dive on that product across regions.
    If not, rank ALL products by revenue change (biggest movers first).
    Powers analyze_products() tool.
    """
    if product:
        product_df = df[df["Product"].str.lower() == product.lower()]
        if product_df.empty:
            return {"error": f"No data found for product '{product}'.", "available_products": sorted(df["Product"].unique().tolist())}

        current, previous = _split_periods(product_df, period_days)
        cur_revenue = float(current["Revenue"].sum())
        prev_revenue = float(previous["Revenue"].sum())
        region_contrib = _contributor_breakdown(current, previous, "Region")

        return {
            "product": product,
            "period_days": period_days,
            "current_revenue": round(cur_revenue, 2),
            "previous_revenue": round(prev_revenue, 2),
            "revenue_change_pct": _pct_change(cur_revenue, prev_revenue),
            "top_region_contributors": region_contrib[:3],
        }
    else:
        current, previous = _split_periods(df, period_days)
        contrib = _contributor_breakdown(current, previous, "Product")
        return {
            "period_days": period_days,
            "all_products_ranked_by_change": contrib,
        }


def get_kpi_summary(df: pd.DataFrame) -> dict:
    """All-time KPI snapshot, used by the Streamlit dashboard headline cards."""
    total_revenue = float(df["Revenue"].sum())
    total_profit = float(df["Profit"].sum())
    total_quantity = float(df["Quantity"].sum())
    margin_pct = round(total_profit / total_revenue * 100, 1) if total_revenue else 0.0

    return {
        "total_revenue": round(total_revenue, 2),
        "total_profit": round(total_profit, 2),
        "total_quantity": round(total_quantity, 2),
        "profit_margin_pct": margin_pct,
        "date_range": {
            "start": df["Date"].min().strftime("%Y-%m-%d"),
            "end": df["Date"].max().strftime("%Y-%m-%d"),
        },
        "num_regions": df["Region"].nunique(),
        "num_products": df["Product"].nunique(),
        "num_records": len(df),
    }
