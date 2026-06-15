"""Load and normalize cafe POS sales data."""

from pathlib import Path

import pandas as pd

# Menu items that are data-quality artifacts, not real products
EXCLUDED_ITEMS = {"error", "unknown"}


def load_sales(csv_path: str | Path) -> pd.DataFrame:
    """Load sales CSV and return normalized daily item quantities."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"item", "quantity", "transaction_date"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    df["item"] = df["item"].str.strip().str.lower()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    df = df.dropna(subset=["quantity", "transaction_date"])
    df = df[~df["item"].isin(EXCLUDED_ITEMS)]

    return df


def daily_item_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per date + menu item."""
    daily = (
        df.groupby([df["transaction_date"].dt.date, "item"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"transaction_date": "date", "quantity": "units_sold"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.dayofweek  # Mon=0
    return daily
