"""
app/data/forecast.py
---------------------
Step: "ML Analysis" (forecasting half) + "predict_next_month()" tool.

Deliberately SIMPLE by design (per the challenge instructions: no heavy
frameworks). We use linear regression on a daily-aggregated revenue series
to project the next N days, plus a naive moving-average baseline for
comparison/sanity-checking. This is easy to explain in an interview:
"it's an ordinary least squares fit on day-index vs revenue."

For a real production system you'd reach for Prophet/ARIMA/etc., but that
would add a heavy dependency for a challenge that explicitly asks to keep
things simple and modular.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from app.config import FORECAST_PERIODS_DAYS
from app.logging_config import setup_logging

logger = setup_logging(__name__)


def _daily_revenue(df: pd.DataFrame, region: str | None = None, product: str | None = None) -> pd.DataFrame:
    filtered = df
    if region:
        filtered = filtered[filtered["Region"] == region]
    if product:
        filtered = filtered[filtered["Product"] == product]

    daily = (
        filtered.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .sort_values("Date")
    )
    return daily


def predict_next_period(
    df: pd.DataFrame,
    periods: int = None,
    region: str | None = None,
    product: str | None = None,
) -> dict:
    """
    Fit a linear trend on daily revenue and project forward `periods` days.

    Returns a dict with:
      - historical_avg_daily_revenue
      - predicted_total_revenue (sum over the forecast horizon)
      - predicted_avg_daily_revenue
      - trend: "increasing" | "decreasing" | "stable"
      - trend_pct: percent change implied by the slope over the horizon
      - r_squared: how well the linear trend fits (0-1, low = noisy data)
    """
    periods = periods or FORECAST_PERIODS_DAYS
    daily = _daily_revenue(df, region, product)

    if len(daily) < 7:
        return {
            "error": "Not enough historical data to forecast (need at least 7 days).",
            "region": region,
            "product": product,
        }

    daily = daily.reset_index(drop=True)
    daily["day_index"] = np.arange(len(daily))

    X = daily[["day_index"]].values
    y = daily["Revenue"].values

    model = LinearRegression()
    model.fit(X, y)
    r_squared = float(model.score(X, y))

    future_idx = np.arange(len(daily), len(daily) + periods).reshape(-1, 1)
    preds = model.predict(future_idx)
    preds = np.clip(preds, a_min=0, a_max=None)  # revenue can't be negative

    historical_avg = float(daily["Revenue"].mean())
    predicted_avg = float(preds.mean())
    predicted_total = float(preds.sum())

    slope = float(model.coef_[0])
    trend_pct = (slope * periods / historical_avg * 100) if historical_avg > 0 else 0.0

    if trend_pct > 2:
        trend = "increasing"
    elif trend_pct < -2:
        trend = "decreasing"
    else:
        trend = "stable"

    result = {
        "region": region or "All Regions",
        "product": product or "All Products",
        "forecast_horizon_days": periods,
        "historical_avg_daily_revenue": round(historical_avg, 2),
        "predicted_avg_daily_revenue": round(predicted_avg, 2),
        "predicted_total_revenue": round(predicted_total, 2),
        "trend": trend,
        "trend_pct": round(trend_pct, 1),
        "r_squared": round(r_squared, 3),
        "confidence_note": (
            "Low confidence (noisy/limited data)" if r_squared < 0.3
            else "Moderate confidence" if r_squared < 0.6
            else "Reasonable confidence"
        ),
    }
    logger.info("Forecast for region=%s product=%s: %s", region, product, result)
    return result
