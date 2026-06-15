"""Convert menu forecasts into ingredient order recommendations."""

from pathlib import Path

import pandas as pd

from src.forecast import DOW_NAMES


def load_recipes(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["menu_item"] = df["menu_item"].str.strip().str.lower()
    return df


def ingredient_demand(forecast: pd.DataFrame, recipes: pd.DataFrame) -> pd.DataFrame:
    """Multiply menu forecasts by recipe BOM to get daily ingredient usage."""
    merged = forecast.merge(
        recipes,
        left_on="item",
        right_on="menu_item",
        how="inner",
    )
    merged["ingredient_qty"] = merged["forecast_units"] * merged["quantity_per_unit"]
    return merged


def weekly_order_list(
    forecast: pd.DataFrame,
    recipes: pd.DataFrame,
    lead_time_days: int = 2,
) -> pd.DataFrame:
    """
    Sum ingredient demand for the week and recommend order quantities.

    lead_time_days models supplier delay (Gnosis-style supply visibility hook).
    """
    demand = ingredient_demand(forecast, recipes)

    weekly = (
        demand.groupby(["ingredient", "unit"], as_index=False)["ingredient_qty"]
        .sum()
        .rename(columns={"ingredient_qty": "weekly_need"})
    )
    weekly["weekly_need"] = weekly["weekly_need"].round(2)
    weekly["recommended_order"] = (weekly["weekly_need"] * 1.05).round(2)  # small rounding buffer
    weekly["lead_time_days"] = lead_time_days
    weekly = weekly.sort_values("weekly_need", ascending=False).reset_index(drop=True)
    return weekly


def format_order_summary(orders: pd.DataFrame) -> str:
    if orders.empty:
        return "No recommendations — check data and recipes."
    lines = ["Weekly ingredient order recommendations:\n"]
    for _, row in orders.iterrows():
        lines.append(
            f"  {row['ingredient']:15} {row['recommended_order']:>8} {row['unit']}"
            f"  (need ~{row['weekly_need']}, {row['lead_time_days']}d lead time)"
        )
    return "\n".join(lines)
