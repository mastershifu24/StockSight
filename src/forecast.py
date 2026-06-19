"""Simple demand forecasting with selectable model."""

import pandas as pd

from src.models import DOW_NAMES, predict

DEFAULT_MODEL = "dow_mean"


def forecast_next_week(
    daily: pd.DataFrame,
    safety_buffer: float = 0.15,
    lookback_weeks: int = 8,
    model: str = DEFAULT_MODEL,
) -> pd.DataFrame:
    """
    Forecast units sold per menu item for each day of the upcoming week.

    Uses the chosen model (default: day-of-week mean) plus optional safety buffer.
    """
    if daily.empty:
        return pd.DataFrame()

    max_date = daily["date"].max()
    next_week_start = max_date + pd.Timedelta(days=1)
    future_dates = pd.date_range(next_week_start, periods=7, freq="D")
    items = daily["item"].unique()

    rows = []
    for d in future_dates:
        for item in items:
            base = predict(model, daily, item, d, lookback_weeks)
            forecast = max(0, round(base * (1 + safety_buffer), 1))
            rows.append(
                {
                    "date": d.date(),
                    "dow": DOW_NAMES[d.dayofweek],
                    "item": item,
                    "forecast_units": forecast,
                    "raw_avg": round(base, 1),
                    "model": model,
                }
            )

    return pd.DataFrame(rows)
