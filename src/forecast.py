"""Simple demand forecasting: day-of-week averages + safety buffer."""

import pandas as pd

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def forecast_next_week(
    daily: pd.DataFrame,
    safety_buffer: float = 0.15,
    lookback_weeks: int = 8,
) -> pd.DataFrame:
    """
    Forecast units sold per menu item for each day of the upcoming week.

    Uses trailing lookback_weeks of day-of-week averages, then applies
    a safety buffer to reduce stockout risk.
    """
    if daily.empty:
        return pd.DataFrame()

    max_date = daily["date"].max()
    cutoff = max_date - pd.Timedelta(weeks=lookback_weeks)
    recent = daily[daily["date"] > cutoff].copy()

    dow_avg = (
        recent.groupby(["item", "dow"], as_index=False)["units_sold"]
        .mean()
        .rename(columns={"units_sold": "avg_units"})
    )

    # Fallback: item overall mean if a DOW has no history
    item_mean = recent.groupby("item", as_index=False)["units_sold"].mean()
    item_mean = item_mean.rename(columns={"units_sold": "fallback_avg"})

    next_week_start = max_date + pd.Timedelta(days=1)
    future_dates = pd.date_range(next_week_start, periods=7, freq="D")

    rows = []
    for d in future_dates:
        dow = d.dayofweek
        for item in recent["item"].unique():
            match = dow_avg[(dow_avg["item"] == item) & (dow_avg["dow"] == dow)]
            if not match.empty:
                base = match["avg_units"].iloc[0]
            else:
                fb = item_mean[item_mean["item"] == item]["fallback_avg"]
                base = fb.iloc[0] if not fb.empty else 0.0

            forecast = max(0, round(base * (1 + safety_buffer), 1))
            rows.append(
                {
                    "date": d.date(),
                    "dow": DOW_NAMES[dow],
                    "item": item,
                    "forecast_units": forecast,
                    "raw_avg": round(base, 1),
                }
            )

    return pd.DataFrame(rows)
