"""Backtesting and model evaluation metrics."""

from __future__ import annotations

import pandas as pd

from src.models import MODEL_NAMES, predict

DEFAULT_HOLDOUT_DAYS = 28
DEFAULT_LOOKBACK_WEEKS = 8


def _safe_mape(actual: float, predicted: float) -> float | None:
    if actual == 0:
        return None
    return abs(actual - predicted) / actual * 100


def backtest_predictions(
    daily: pd.DataFrame,
    model: str,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
) -> pd.DataFrame:
    """
    Walk-forward backtest on the last holdout_days.

    For each (date, item) in the holdout window, train only on prior dates
    and record predicted vs actual units sold.
    """
    if daily.empty:
        return pd.DataFrame()

    daily = daily.sort_values("date").copy()
    max_date = daily["date"].max()
    holdout_start = max_date - pd.Timedelta(days=holdout_days - 1)
    holdout = daily[daily["date"] >= holdout_start]
    items = daily["item"].unique()

    rows = []
    for _, row in holdout.iterrows():
        pred = predict(model, daily, row["item"], row["date"], lookback_weeks)
        pred = max(0.0, pred)
        actual = float(row["units_sold"])
        rows.append(
            {
                "date": row["date"],
                "item": row["item"],
                "dow": row["dow"],
                "actual": actual,
                "predicted": round(pred, 2),
                "error": round(actual - pred, 2),
                "abs_error": abs(actual - pred),
                "model": model,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics(preds: pd.DataFrame, safety_buffer: float = 0.15) -> dict:
    """Aggregate MAE, MAPE, RMSE, and stockout-proxy rate."""
    if preds.empty:
        return {}

    mae = preds["abs_error"].mean()
    rmse = (preds["error"] ** 2).mean() ** 0.5

    mapes = [_safe_mape(a, p) for a, p in zip(preds["actual"], preds["predicted"])]
    mapes = [m for m in mapes if m is not None]
    mape = sum(mapes) / len(mapes) if mapes else None

    # Stockout proxy: actual demand exceeded forecast (manager runs out)
    unbuffered_stockouts = (preds["actual"] > preds["predicted"]).mean() * 100
    buffered = preds["predicted"] * (1 + safety_buffer)
    buffered_stockouts = (preds["actual"] > buffered).mean() * 100

    return {
        "n_predictions": len(preds),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mape_pct": round(mape, 2) if mape is not None else None,
        "stockout_rate_pct": round(unbuffered_stockouts, 1),
        "stockout_rate_buffered_pct": round(buffered_stockouts, 1),
    }


def compare_models(
    daily: pd.DataFrame,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    lookback_weeks: int = DEFAULT_LOOKBACK_WEEKS,
    safety_buffer: float = 0.15,
) -> pd.DataFrame:
    """Run backtest for each model and return comparison table."""
    results = []
    for model in MODEL_NAMES:
        preds = backtest_predictions(daily, model, holdout_days, lookback_weeks)
        metrics = summarize_metrics(preds, safety_buffer)
        if metrics:
            results.append({"model": model, **metrics})
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("mae").reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
    return df


def select_best_model(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "dow_mean"
    return str(comparison.iloc[0]["model"])
