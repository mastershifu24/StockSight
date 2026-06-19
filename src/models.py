"""Forecast model implementations for StockSight."""

from __future__ import annotations

import pandas as pd

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

MODEL_NAMES = ("dow_mean", "rolling_7", "global_mean", "exp_smoothing")


def _history_before(history: pd.DataFrame, before_date: pd.Timestamp, lookback_weeks: int) -> pd.DataFrame:
    cutoff = before_date - pd.Timedelta(weeks=lookback_weeks)
    recent = history[(history["date"] < before_date) & (history["date"] > cutoff)]
    if recent.empty:
        recent = history[history["date"] < before_date]
    return recent


def predict_global_mean(history: pd.DataFrame, item: str) -> float:
    subset = history[history["item"] == item]["units_sold"]
    return float(subset.mean()) if not subset.empty else 0.0


def predict_dow_mean(history: pd.DataFrame, item: str, dow: int) -> float:
    subset = history[(history["item"] == item) & (history["dow"] == dow)]["units_sold"]
    if not subset.empty:
        return float(subset.mean())
    return predict_global_mean(history, item)


def predict_rolling_mean(history: pd.DataFrame, item: str, window: int = 7) -> float:
    subset = history[history["item"] == item].sort_values("date")
    if subset.empty:
        return 0.0
    return float(subset["units_sold"].tail(window).mean())


def predict_exp_smoothing(history: pd.DataFrame, item: str, alpha: float = 0.3) -> float:
    subset = history[history["item"] == item].sort_values("date")["units_sold"]
    if subset.empty:
        return 0.0
    level = float(subset.iloc[0])
    for value in subset.iloc[1:]:
        level = alpha * float(value) + (1 - alpha) * level
    return level


def predict(
    model: str,
    history: pd.DataFrame,
    item: str,
    target_date: pd.Timestamp,
    lookback_weeks: int = 8,
) -> float:
    """Point forecast for one item on one date using only prior history."""
    hist = _history_before(history, target_date, lookback_weeks)
    if model == "global_mean":
        return predict_global_mean(hist, item)
    if model == "dow_mean":
        return predict_dow_mean(hist, item, target_date.dayofweek)
    if model == "rolling_7":
        return predict_rolling_mean(hist, item, window=7)
    if model == "exp_smoothing":
        return predict_exp_smoothing(hist, item, alpha=0.3)
    raise ValueError(f"Unknown model: {model}")
