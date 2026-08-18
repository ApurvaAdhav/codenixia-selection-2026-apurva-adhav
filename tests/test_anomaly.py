"""
tests/test_anomaly.py
-----------------------
Tests app/data/anomaly.py. Uses a bigger synthetic series (not the small
sample_df fixture) because Isolation Forest needs enough points per group
to fit meaningfully.
"""
import numpy as np
import pandas as pd
import pytest

from app.data.anomaly import detect_anomalies


@pytest.fixture
def anomaly_df() -> pd.DataFrame:
    """90 days of stable revenue for Region North / Product X, with a
    sharp, sustained drop injected into the last 20 days."""
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    rows = []
    rng = np.random.default_rng(0)
    for i, d in enumerate(dates):
        base_revenue = 1000.0
        noise = rng.normal(0, 20)  # small noise
        if i >= 70:  # last 20 days: sharp drop
            revenue = base_revenue * 0.3 + noise
        else:
            revenue = base_revenue + noise
        rows.append(dict(
            Date=d, Region="North", Product="Product X", Category="Cat",
            Customer_Segment="Consumer", Quantity=10, Revenue=max(revenue, 0),
            Cost=500.0, Profit=max(revenue, 0) - 500.0,
        ))
    return pd.DataFrame(rows)


def test_detect_anomalies_finds_the_injected_drop(anomaly_df):
    anomalies = detect_anomalies(anomaly_df)
    assert len(anomalies) > 0

    # At least one flagged anomaly should be in the injected drop window
    # (last 20 days) and flagged as "below normal".
    drop_dates = set(pd.date_range("2024-02-10", "2024-03-30").strftime("%Y-%m-%d"))
    below_normal_in_window = [
        a for a in anomalies if a["direction"] == "below normal" and a["date"] in drop_dates
    ]
    assert len(below_normal_in_window) > 0


def test_detect_anomalies_empty_df_returns_empty():
    empty = pd.DataFrame(columns=["Date", "Region", "Product", "Revenue", "Quantity"])
    assert detect_anomalies(empty) == []


def test_detect_anomalies_skips_tiny_groups():
    tiny = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3),
        "Region": ["North"] * 3,
        "Product": ["X"] * 3,
        "Revenue": [100.0, 110.0, 90.0],
        "Quantity": [1, 1, 1],
    })
    # Too few points (<15) for a meaningful model -> should not error, just skip.
    assert detect_anomalies(tiny) == []
