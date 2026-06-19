"""Run model backtest and save evaluation results."""

import argparse
from pathlib import Path

import pandas as pd

from src.evaluate import (
    DEFAULT_HOLDOUT_DAYS,
    DEFAULT_LOOKBACK_WEEKS,
    backtest_predictions,
    compare_models,
    select_best_model,
)
from src.ingest import daily_item_sales, load_sales
from src.menu import apply_demo_menu_labels, load_demo_item_map

ROOT = Path(__file__).parent
DEFAULT_SALES = ROOT / "Cleaned_DataSet.csv"
OUT_COMPARISON = ROOT / "data" / "model_comparison.csv"
OUT_PREDICTIONS = ROOT / "data" / "model_backtest_predictions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="StockSight model backtest")
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    parser.add_argument("--holdout-days", type=int, default=DEFAULT_HOLDOUT_DAYS)
    parser.add_argument("--lookback-weeks", type=int, default=DEFAULT_LOOKBACK_WEEKS)
    parser.add_argument("--buffer", type=float, default=0.15)
    parser.add_argument("--no-demo-map", action="store_true")
    args = parser.parse_args()

    sales = load_sales(args.sales)
    if not args.no_demo_map:
        sales = apply_demo_menu_labels(sales, load_demo_item_map())
    daily = daily_item_sales(sales)

    comparison = compare_models(
        daily,
        holdout_days=args.holdout_days,
        lookback_weeks=args.lookback_weeks,
        safety_buffer=args.buffer,
    )
    best = select_best_model(comparison)

    comparison.to_csv(OUT_COMPARISON, index=False)
    best_preds = backtest_predictions(
        daily, best, args.holdout_days, args.lookback_weeks
    )
    best_preds.to_csv(OUT_PREDICTIONS, index=False)

    print(f"Backtest: last {args.holdout_days} days | lookback {args.lookback_weeks} weeks")
    print(f"Items: {daily['item'].nunique()} | rows: {len(daily):,}\n")
    print("Model comparison (sorted by MAE):\n")
    print(comparison.to_string(index=False))
    print(f"\nBest model by MAE: {best}")
    print(f"Saved: {OUT_COMPARISON.name}, {OUT_PREDICTIONS.name}")


if __name__ == "__main__":
    main()
