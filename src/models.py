"""Forecast model implementations for StockSight."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

MODEL_NAMES = ("dow_mean", "rolling_7", "global_mean", "exp_smoothing", "ridge_regression")

FEATURE_COLS = ("dow", "lag_1", "lag_7")
MIN_RIDGE_ROWS = 14
RIDGE_ALPHA = 1.0


def _history_before(history: pd.DataFrame, before_date: pd.Timestamp, lookback_weeks: int) -> pd.DataFrame:
    cutoff = before_date - pd.Timedelta(weeks=lookback_weeks)
    recent = history[(history["date"] < before_date) & (history["date"] > cutoff)]
    if recent.empty:
        recent = history[history["date"] < before_date]
    return recent


def _item_series(history: pd.DataFrame, item: str) -> pd.DataFrame:
    return history[history["item"] == item].sort_values("date").copy()


def _units_on_or_before(series: pd.DataFrame, day: pd.Timestamp) -> float | None:
    match = series[series["date"] == day]
    if not match.empty:
        return float(match.iloc[-1]["units_sold"])
    prior = series[series["date"] < day]
    if prior.empty:
        return None
    return float(prior.iloc[-1]["units_sold"])


def _ridge_training_frame(series: pd.DataFrame) -> pd.DataFrame:
    """Build supervised rows: dow + lags -> next-day units sold."""
    frame = series.copy()
    frame["lag_1"] = frame["units_sold"].shift(1)
    frame["lag_7"] = frame["units_sold"].shift(7)
    return frame.dropna(subset=["lag_1", "lag_7"])


def _ridge_features(series: pd.DataFrame, target_date: pd.Timestamp) -> list[float] | None:
    lag_1_day = target_date - pd.Timedelta(days=1)
    lag_7_day = target_date - pd.Timedelta(days=7)
    lag_1 = _units_on_or_before(series, lag_1_day)
    lag_7 = _units_on_or_before(series, lag_7_day)
    if lag_1 is None or lag_7 is None:
        return None
    return [float(target_date.dayofweek), lag_1, lag_7]


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


def predict_ridge(history: pd.DataFrame, item: str, target_date: pd.Timestamp) -> float:
    """
    Ridge regression on lag features (dow, lag_1, lag_7).

    Retrained on each prediction using only prior history for that item.
    Falls back to day-of-week mean when history is too short.
    """
    series = _item_series(history, item)
    train = _ridge_training_frame(series)
    if len(train) < MIN_RIDGE_ROWS:
        return predict_dow_mean(history, item, target_date.dayofweek)

    features = _ridge_features(series, target_date)
    if features is None:
        return predict_dow_mean(history, item, target_date.dayofweek)

    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(train[list(FEATURE_COLS)], train["units_sold"])
    x_pred = pd.DataFrame([features], columns=list(FEATURE_COLS))
    pred = float(model.predict(x_pred)[0])
    return max(0.0, pred)


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
    if model == "ridge_regression":
        return predict_ridge(hist, item, target_date)
    raise ValueError(f"Unknown model: {model}")
