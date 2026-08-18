"""
tests/test_forecast.py
------------------------
Tests app/data/forecast.py: linear trend forecasting.
"""
import pandas as pd
import pytest

from app.data.forecast import predict_next_period


@pytest.fixture
def trending_df() -> pd.DataFrame:
    """A perfectly linear upward trend: revenue = 100 * day_index, so we
    can assert the forecast direction and rough magnitude precisely."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    rows = []
    for i, d in enumerate(dates):
        revenue = 1000 + i * 50  # clean upward trend
        rows.append(dict(
            Date=d, Region="North", Product="Product X", Category="Cat",
            Customer_Segment="Consumer", Quantity=10, Revenue=revenue,
            Cost=500.0, Profit=revenue - 500.0,
        ))
    return pd.DataFrame(rows)


def test_forecast_detects_increasing_trend(trending_df):
    result = predict_next_period(trending_df, periods=30)
    assert result["trend"] == "increasing"
    assert result["trend_pct"] > 0
    assert result["r_squared"] > 0.9  # near-perfect linear data


def test_forecast_insufficient_data_returns_error():
    tiny_df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=3),
        "Region": ["North"] * 3,
        "Product": ["X"] * 3,
        "Revenue": [100.0, 110.0, 105.0],
    })
    result = predict_next_period(tiny_df, periods=30)
    assert "error" in result


def test_forecast_scoped_to_region_and_product(trending_df):
    result = predict_next_period(trending_df, periods=10, region="North", product="Product X")
    assert result["region"] == "North"
    assert result["product"] == "Product X"


def test_forecast_predicted_revenue_never_negative(trending_df):
    # Even for a declining series, predicted revenue should be clipped at 0.
    declining = trending_df.copy()
    declining["Revenue"] = 1000 - declining.index * 100  # will go negative fast
    result = predict_next_period(declining, periods=30)
    assert result["predicted_total_revenue"] >= 0
