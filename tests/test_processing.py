"""
tests/test_processing.py
--------------------------
Tests app/data/processing.py: column validation, cleaning, dedup, and
profit recomputation.
"""
import pandas as pd
import pytest

from app.data.processing import (
    DataValidationError, clean_sales_data, load_and_clean, validate_columns,
)


def test_messy_csv_is_cleaned_correctly(messy_csv_bytes):
    df, report = load_and_clean(messy_csv_bytes, "messy.csv")

    # 5 rows in -> 1 duplicate removed, 2 rows dropped for missing essentials
    # (missing Date, missing Quantity) -> 2 valid rows remain.
    assert report.rows_in == 5
    assert report.dropped_duplicates == 1
    assert report.dropped_missing_required == 2
    assert report.rows_out == 2
    assert len(df) == 2

    # The row with missing Profit should have had it recomputed as Revenue-Cost.
    recomputed_row = df[(df["Date"] == "2024-01-02")].iloc[0]
    assert recomputed_row["Profit"] == pytest.approx(200.0)  # 500 - 300
    assert report.profit_recomputed == 1


def test_missing_required_columns_raises():
    bad_df = pd.DataFrame({"Date": ["2024-01-01"], "Region": ["North"]})
    with pytest.raises(DataValidationError):
        validate_columns(bad_df)


def test_unsupported_file_extension_raises():
    from app.data.processing import load_dataframe
    with pytest.raises(DataValidationError):
        load_dataframe(b"not a real file", "data.txt")


def test_all_rows_invalid_raises():
    csv_text = (
        "Date,Region,Product,Category,Customer_Segment,Quantity,Revenue,Cost,Profit\n"
        ",North,Product A,Electronics,Consumer,10,1000,600,400\n"
    ).encode("utf-8")
    with pytest.raises(DataValidationError):
        load_and_clean(csv_text, "allbad.csv")


def test_column_aliases_are_normalized():
    """Real-world uploads might use slightly different column names/casing -
    make sure common variants still validate."""
    df = pd.DataFrame({
        "date": ["2024-01-01"], "region": ["North"], "product": ["A"],
        "category": ["Cat"], "customer segment": ["Consumer"],
        "quantity": [1], "revenue": [10.0], "cost": [5.0], "profit": [5.0],
    })
    validate_columns(df)  # should not raise


def test_negative_values_are_dropped():
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "Region": ["North", "North"],
        "Product": ["A", "A"],
        "Category": ["Cat", "Cat"],
        "Customer_Segment": ["Consumer", "Consumer"],
        "Quantity": [10, -5],
        "Revenue": [100.0, -50.0],
        "Cost": [50.0, 20.0],
        "Profit": [50.0, -70.0],
    })
    clean_df, report = clean_sales_data(df)
    assert len(clean_df) == 1
    assert "negative" in report.notes[0].lower()
