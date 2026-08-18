"""
app/data/processing.py
-----------------------
Step 1 of the pipeline: "CSV/Excel Data -> Python Data Processing".

Responsibilities:
  1. Load a CSV or Excel file into a DataFrame.
  2. Validate that required columns exist.
  3. Clean: parse dates, coerce numeric types, drop bad rows, recompute
     Profit if missing/inconsistent, remove duplicates.
  4. Return a clean DataFrame + a report describing what was done (useful
     for both logging and showing the user what happened to their data).

Keeping this pure-function / no-side-effects (besides logging) makes it
trivial to unit test.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

from app.logging_config import setup_logging

logger = setup_logging(__name__)

REQUIRED_COLUMNS = [
    "Date", "Region", "Product", "Category",
    "Customer_Segment", "Quantity", "Revenue", "Cost", "Profit",
]

# Accept common alternate spellings/casing from real-world uploads.
COLUMN_ALIASES = {
    "customer segment": "Customer_Segment",
    "customersegment": "Customer_Segment",
    "customer_segment": "Customer_Segment",
    "date": "Date",
    "region": "Region",
    "product": "Product",
    "category": "Category",
    "quantity": "Quantity",
    "qty": "Quantity",
    "revenue": "Revenue",
    "sales": "Revenue",
    "cost": "Cost",
    "profit": "Profit",
}


class DataValidationError(Exception):
    """Raised when the uploaded dataset cannot be used, even after cleaning."""


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped_missing_required: int = 0
    dropped_duplicates: int = 0
    profit_recomputed: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "rows_in": self.rows_in,
            "rows_out": self.rows_out,
            "dropped_missing_required": self.dropped_missing_required,
            "dropped_duplicates": self.dropped_duplicates,
            "profit_recomputed": self.profit_recomputed,
            "notes": self.notes,
        }


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_").replace("-", "_")
        # try direct alias match first, then a normalized alias match
        if col.strip().lower() in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[col.strip().lower()]
        elif key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
        else:
            rename_map[col] = col.strip()
    return df.rename(columns=rename_map)


def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel bytes into a DataFrame based on file extension."""
    lower = filename.lower()
    try:
        if lower.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(file_bytes))
        elif lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            raise DataValidationError(
                f"Unsupported file type for '{filename}'. Use .csv or .xlsx."
            )
    except DataValidationError:
        raise
    except Exception as exc:
        logger.exception("Failed to parse uploaded file")
        raise DataValidationError(f"Could not parse '{filename}': {exc}") from exc

    return df


def validate_columns(df: pd.DataFrame) -> None:
    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Dataset is missing required columns: {missing}. "
            f"Required columns are: {REQUIRED_COLUMNS}"
        )


def clean_sales_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Full cleaning pipeline. Returns (clean_df, report).
    Raises DataValidationError if the data cannot be salvaged.
    """
    report = CleaningReport(rows_in=len(df))

    df = _normalize_columns(df)
    validate_columns(df)

    # --- Parse types -------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    numeric_cols = ["Quantity", "Revenue", "Cost", "Profit"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["Region", "Product", "Category", "Customer_Segment"]:
        df[col] = df[col].astype(str).str.strip()

    # --- Drop rows missing anything essential -------------------------
    before = len(df)
    essential = ["Date", "Region", "Product", "Quantity", "Revenue"]
    df = df.dropna(subset=essential)
    report.dropped_missing_required = before - len(df)

    if df.empty:
        raise DataValidationError(
            "After cleaning, no valid rows remained. Check date formats and "
            "that Quantity/Revenue are numeric."
        )

    # --- Recompute Profit where missing or inconsistent ----------------
    # If Cost is missing, assume Cost = Revenue - Profit when Profit exists,
    # else 0. If Profit is missing/NaN, recompute from Revenue - Cost.
    missing_cost = df["Cost"].isna()
    missing_profit = df["Profit"].isna()

    df.loc[missing_cost & ~missing_profit, "Cost"] = (
        df.loc[missing_cost & ~missing_profit, "Revenue"]
        - df.loc[missing_cost & ~missing_profit, "Profit"]
    )
    df["Cost"] = df["Cost"].fillna(0)

    recompute_mask = missing_profit
    df.loc[recompute_mask, "Profit"] = (
        df.loc[recompute_mask, "Revenue"] - df.loc[recompute_mask, "Cost"]
    )
    report.profit_recomputed = int(recompute_mask.sum())

    # --- Remove exact duplicate rows -----------------------------------
    before = len(df)
    df = df.drop_duplicates()
    report.dropped_duplicates = before - len(df)

    # --- Drop negative quantities / revenue (bad data, not returns) ----
    bad_rows = (df["Quantity"] < 0) | (df["Revenue"] < 0)
    if bad_rows.any():
        report.notes.append(
            f"Dropped {int(bad_rows.sum())} rows with negative Quantity/Revenue."
        )
        df = df[~bad_rows]

    df = df.sort_values("Date").reset_index(drop=True)
    report.rows_out = len(df)

    logger.info(
        "Cleaned data: %s -> %s rows (dropped_missing=%s, dupes=%s, profit_recomputed=%s)",
        report.rows_in, report.rows_out, report.dropped_missing_required,
        report.dropped_duplicates, report.profit_recomputed,
    )

    return df, report


def load_and_clean(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, CleaningReport]:
    """Convenience wrapper: load + clean in one call."""
    df = load_dataframe(file_bytes, filename)
    return clean_sales_data(df)
