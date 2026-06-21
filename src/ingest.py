"""Load and normalize cafe POS sales data."""

from __future__ import annotations

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
        df.assign(sale_date=df["transaction_date"].dt.date)
        .groupby(["sale_date", "item"], as_index=False)["quantity"]
        .sum()
        .rename(columns={"sale_date": "date", "quantity": "units_sold"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.dayofweek  # Mon=0
    return daily


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    csv = root / "Cleaned_DataSet.csv"
    print(f"Reading: {csv}\n")

    sales = load_sales(csv)
    daily = daily_item_sales(sales)

    print(f"Transactions: {len(sales):,}")
    print(f"Date range: {sales['transaction_date'].min().date()} → {sales['transaction_date'].max().date()}")
    print(f"Daily rows: {len(daily):,}\n")
    print(daily.head(8).to_string(index=False))
