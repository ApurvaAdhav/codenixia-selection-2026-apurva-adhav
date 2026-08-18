"""
app/data/anomaly.py
--------------------
Step: "ML Analysis" (anomaly detection half).

Uses scikit-learn's IsolationForest to flag unusual daily sales records at
the (Region, Product) grain. We aggregate to daily revenue per Region+Product
first, because row-level anomaly detection on raw transactions is noisy -
what a business manager actually cares about is "this region/product's daily
revenue pattern looks abnormal."

Function returns plain Python data structures (list of dicts) so it is easy
to serialize to JSON for the API and easy to feed into the LLM prompt.
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config import ANOMALY_CONTAMINATION
from app.logging_config import setup_logging

logger = setup_logging(__name__)

# How far back (from the most recent date in the data) we consider an
# anomaly "current" and worth surfacing to a business manager. The model
# is still FIT on the full history (so it learns each series' normal
# range), but we only REPORT anomalies inside this recency window. This
# is what makes detect_anomalies() answer "what's wrong right now" rather
# than surfacing random noisy outlier days from a year ago.
RECENT_WINDOW_DAYS = 60


def _build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to Date x Region x Product daily revenue/quantity."""
    grouped = (
        df.groupby(["Date", "Region", "Product"], as_index=False)
        .agg(Revenue=("Revenue", "sum"), Quantity=("Quantity", "sum"))
    )
    return grouped


def detect_anomalies(df: pd.DataFrame, contamination: float = None) -> list[dict]:
    """
    Run Isolation Forest per (Region, Product) group to find days whose
    revenue deviates unusually from that series' own recent trend.

    Two design choices worth knowing for an interview walkthrough:

    1. DETRENDING: raw daily revenue naturally trends upward over a year
       and has weekly seasonality (weekday vs weekend). Feeding raw levels
       into Isolation Forest confuses "this is just December, revenue is
       naturally higher" with a true anomaly. So we first compute each
       day's REsidual = (actual - 14-day trailing rolling average) / rolling
       average, i.e. "% deviation from its own recent baseline", and run
       Isolation Forest on that residual series instead of raw revenue.
       This makes a sustained real shift (e.g. a 50% demand drop) stand out
       clearly, regardless of overall trend/seasonality.
    2. PER-GROUP MODELING: we fit one small model per (Region, Product)
       pair rather than one global model, so "Product A is just bigger
       than Product E" is never confused with an actual anomaly.

    Returns a list of anomaly records (recent window only - see
    RECENT_WINDOW_DAYS), sorted by how anomalous they are (most negative
    anomaly_score first = most anomalous).
    """
    if df.empty:
        return []

    contamination = contamination or ANOMALY_CONTAMINATION
    daily = _build_daily_series(df)
    max_date = daily["Date"].max()
    recent_cutoff = max_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)

    anomalies = []
    for (region, product), group in daily.groupby(["Region", "Product"]):
        if len(group) < 15:
            # Not enough data points for a meaningful rolling baseline + model.
            continue

        group = group.sort_values("Date").reset_index(drop=True)

        # Trailing rolling average = "expected" revenue for that day, based
        # on the series' own recent history (adapts to trend automatically).
        rolling_baseline = group["Revenue"].rolling(window=14, min_periods=7).mean()
        group["baseline"] = rolling_baseline
        group["residual_pct"] = (
            (group["Revenue"] - group["baseline"]) / group["baseline"] * 100
        )
        group = group.dropna(subset=["residual_pct"]).reset_index(drop=True)
        if len(group) < 10:
            continue

        X = group[["residual_pct"]].values

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        preds = model.fit_predict(X)          # -1 = anomaly, 1 = normal
        scores = model.decision_function(X)   # lower = more anomalous

        group["is_anomaly"] = preds == -1
        group["anomaly_score"] = scores

        recent_group = group[group["Date"] >= recent_cutoff]
        for _, row in recent_group[recent_group["is_anomaly"]].iterrows():
            anomalies.append({
                "date": row["Date"].strftime("%Y-%m-%d"),
                "region": region,
                "product": product,
                "revenue": round(float(row["Revenue"]), 2),
                "expected_revenue": round(float(row["baseline"]), 2),
                "pct_vs_expected": round(float(row["residual_pct"]), 1),
                "anomaly_score": round(float(row["anomaly_score"]), 4),
                "direction": "below normal" if row["residual_pct"] < 0 else "above normal",
            })

    anomalies.sort(key=lambda a: a["anomaly_score"])
    logger.info(
        "Detected %d anomalies across %d region/product groups",
        len(anomalies), daily.groupby(["Region", "Product"]).ngroups,
    )
    return anomalies


def summarize_anomalies(anomalies: list[dict], top_n: int = 5) -> str:
    """Human-readable one-liner summary, used as a fallback / quick text."""
    if not anomalies:
        return "No significant anomalies detected."
    top = anomalies[:top_n]
    parts = [
        f"{a['product']} in {a['region']} on {a['date']} "
        f"({a['direction']}, {a['pct_vs_expected']:+.1f}% vs its expected trend)"
        for a in top
    ]
    return "Anomalies found: " + "; ".join(parts)
