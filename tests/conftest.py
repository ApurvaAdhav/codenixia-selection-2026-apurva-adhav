"""
tests/conftest.py
-------------------
Shared pytest fixtures. Building a small synthetic DataFrame here (rather
than loading the full 14k-row sample CSV in every test) keeps the test
suite fast and makes each test's expected numbers easy to hand-verify.
"""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """
    A small, hand-crafted dataset spanning 60 days, with a KNOWN, exact
    drop: Region West / Product A revenue is exactly halved in the most
    recent 30 days vs the previous 30 days. This lets tests assert precise
    expected percentages instead of loose bounds.
    """
    dates_p1 = pd.date_range("2024-01-01", periods=30, freq="D")   # previous period
    dates_p2 = pd.date_range("2024-01-31", periods=30, freq="D")   # current period

    rows = []
    for d in dates_p1:
        rows.append(dict(Date=d, Region="West", Product="Product A", Category="Electronics",
                          Customer_Segment="Consumer", Quantity=10, Revenue=1000.0, Cost=600.0, Profit=400.0))
        rows.append(dict(Date=d, Region="East", Product="Product B", Category="Apparel",
                          Customer_Segment="Corporate", Quantity=5, Revenue=500.0, Cost=300.0, Profit=200.0))
    for d in dates_p2:
        # West/Product A revenue halved in the current period (known anomaly).
        rows.append(dict(Date=d, Region="West", Product="Product A", Category="Electronics",
                          Customer_Segment="Consumer", Quantity=5, Revenue=500.0, Cost=300.0, Profit=200.0))
        rows.append(dict(Date=d, Region="East", Product="Product B", Category="Apparel",
                          Customer_Segment="Corporate", Quantity=5, Revenue=500.0, Cost=300.0, Profit=200.0))

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@pytest.fixture
def messy_csv_bytes() -> bytes:
    """CSV bytes with missing values, a duplicate row, and a missing Profit
    column entry - used to test app/data/processing.py cleaning logic."""
    csv_text = (
        "Date,Region,Product,Category,Customer_Segment,Quantity,Revenue,Cost,Profit\n"
        "2024-01-01,North,Product A,Electronics,Consumer,10,1000,600,400\n"
        "2024-01-01,North,Product A,Electronics,Consumer,10,1000,600,400\n"  # duplicate
        "2024-01-02,North,Product A,Electronics,Consumer,5,500,300,\n"       # missing profit -> recompute
        ",South,Product B,Apparel,Corporate,3,300,150,150\n"                  # missing date -> dropped
        "2024-01-03,South,Product B,Apparel,Corporate,,300,150,150\n"        # missing qty -> dropped
    )
    return csv_text.encode("utf-8")
