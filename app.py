"""StockSight Streamlit demo — Think Coffee tailored UI."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.evaluate import compare_models
from src.forecast import DOW_NAMES, forecast_next_week
from src.ingest import daily_item_sales, load_sales
from src.menu import (
    apply_demo_menu_labels,
    item_display_names,
    load_demo_item_map,
    load_menu_catalog,
    load_yelp_popular,
)
from src.recommend import load_recipes, weekly_order_list
from src.ui import show_dataframe, show_plotly

ROOT = Path(__file__).parent
DEFAULT_SALES = ROOT / "Cleaned_DataSet.csv"
DEFAULT_RECIPES = ROOT / "data" / "recipes.csv"

st.set_page_config(page_title="StockSight | Think Coffee", page_icon="☕", layout="wide")
st.title("StockSight")
st.caption("Weekly ingredient orders for Think Coffee — demo until your POS export is plugged in.")

with st.sidebar:
    st.header("Settings")
    think_coffee_demo = st.toggle(
        "Think Coffee demo labels",
        value=True,
        help="Maps bundled demo sales to Think Coffee menu items and recipes.",
    )
    safety_buffer = st.slider("Safety buffer", 0.0, 0.5, 0.15, 0.05)
    lookback_weeks = st.slider("Lookback weeks", 2, 16, 8)
    lead_time_days = st.slider("Supplier lead time (days)", 1, 7, 2)
    uploaded = st.file_uploader("Upload POS CSV (Think Coffee export)", type=["csv"])

@st.cache_data
def _load_default(path: str) -> pd.DataFrame:
    return load_sales(path)


def _yelp_api_key() -> str | None:
    try:
        key = st.secrets.get("YELP_API_KEY", "")
        return key.strip() if key else None
    except Exception:
        return None


@st.cache_data(ttl=604800, show_spinner=False)
def _load_yelp_reviews(api_key: str | None) -> pd.DataFrame:
    from src.yelp_data import get_reviews

    return get_reviews(api_key)


@st.cache_data(show_spinner=False)
def _load_yelp_locations() -> pd.DataFrame:
    from src.yelp_data import load_bundled_locations

    return load_bundled_locations()


@st.cache_data(show_spinner="Evaluating forecast models...")
def _model_comparison(_daily: pd.DataFrame, holdout: int, lookback: int, buffer: float) -> pd.DataFrame:
    return compare_models(_daily, holdout_days=holdout, lookback_weeks=lookback, safety_buffer=buffer)


item_map = load_demo_item_map()
yelp_popular = load_yelp_popular()
yelp_reviews = _load_yelp_reviews(_yelp_api_key())
yelp_locations = _load_yelp_locations()
menu_catalog = load_menu_catalog()

if uploaded:
    sales = load_sales(uploaded)
    data_label = uploaded.name
    using_demo_map = False
else:
    sales = _load_default(str(DEFAULT_SALES))
    data_label = "Bundled cafe sales (demo)"
    using_demo_map = think_coffee_demo

if using_demo_map:
    sales = apply_demo_menu_labels(sales, item_map)

daily = daily_item_sales(sales)
display_names = item_display_names(daily, item_map) if using_demo_map else {
    i: i.replace("_", " ").title() for i in daily["item"].unique()
}

recipes = load_recipes(DEFAULT_RECIPES)
forecast = forecast_next_week(daily, safety_buffer=safety_buffer, lookback_weeks=lookback_weeks)
orders = weekly_order_list(forecast, recipes, lead_time_days=lead_time_days)

# Human-friendly labels on forecast
forecast_display = forecast.copy()
forecast_display["item_label"] = forecast_display["item"].map(display_names).fillna(forecast_display["item"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Transactions", f"{len(sales):,}")
col2.metric("Date range", f"{daily['date'].min().date()} to {daily['date'].max().date()}")
col3.metric("Menu items", daily["item"].nunique())
col4.metric("Ingredients tracked", orders["ingredient"].nunique())

st.subheader("Weekly ingredient orders")
show_dataframe(orders, hide_index=True)

if "bacon" in orders["ingredient"].values:
    bacon_row = orders[orders["ingredient"] == "bacon"].iloc[0]
    st.info(
        f"**Bacon:** order ~{bacon_row['recommended_order']} {bacon_row['unit']} this week "
        f"(from egg sandwiches, ranchero, burritos). Matches the stockout problem your barista mentioned."
    )

st.subheader("Menu forecast (next 7 days)")
pivot = (
    forecast_display.pivot_table(index="item_label", columns="dow", values="forecast_units", aggfunc="sum")
    .reindex(columns=DOW_NAMES)
    .fillna(0)
)
show_dataframe(pivot)

chart_df = daily.groupby("item", as_index=False)["units_sold"].sum()
chart_df["item_label"] = chart_df["item"].map(display_names)
chart_df = chart_df.sort_values("units_sold", ascending=True)
fig = px.bar(
    chart_df,
    x="units_sold",
    y="item_label",
    orientation="h",
    title="Historical demand (Think Coffee menu labels in demo mode)",
)
show_plotly(fig)

with st.expander("Model evaluation (data science)"):
    st.caption(
        "28-day holdout backtest. Each model predicts daily sales using only prior history. "
        "**Production uses day-of-week mean** for interpretability; re-run backtest on real POS to pick the winner."
    )
    comparison = _model_comparison(daily, holdout=28, lookback=lookback_weeks, buffer=safety_buffer)
    show_dataframe(comparison, hide_index=True)
    if not comparison.empty:
        best_mae = comparison.iloc[0]["model"]
        dow_row = comparison[comparison["model"] == "dow_mean"]
        if not dow_row.empty:
            st.write(
                f"Lowest MAE on holdout: **{best_mae}** | "
                f"dow_mean stockout rate: **{dow_row.iloc[0]['stockout_rate_pct']:.1f}%** "
                f"→ with {safety_buffer:.0%} buffer: **{dow_row.iloc[0]['stockout_rate_buffered_pct']:.1f}%**"
            )
        else:
            st.write(f"Lowest MAE on holdout: **{best_mae}**")
    st.caption("Full notebook: `notebooks/stocksight_eda_backtest.ipynb` | CLI: `python run_backtest.py`")

with st.expander("Yelp demand signals (public reviews)"):
    store_n = (
        yelp_reviews["store_address"].nunique()
        if "store_address" in yelp_reviews.columns and not yelp_reviews.empty
        else 1
    )
    total_stores = len(yelp_locations) if not yelp_locations.empty else store_n
    st.caption(
        f"**{len(yelp_reviews)} reviews** across **{store_n} locations** "
        f"(Think Coffee has **{total_stores}** NYC stores). "
        "Popular drinks from public Yelp data. **Not used for order forecasting.**"
    )
    if not yelp_locations.empty:
        st.markdown("**All NYC locations**")
        show_dataframe(yelp_locations, hide_index=True)
    show_dataframe(yelp_popular.sort_values("review_mentions", ascending=False), hide_index=True)
    if not yelp_reviews.empty:
        st.markdown("**Sample reviews by location**")
        review_cols = [
            c for c in ["neighborhood", "store_address", "user", "rating", "date", "menu_mentions"]
            if c in yelp_reviews.columns
        ]
        show_dataframe(yelp_reviews[review_cols], hide_index=True)

with st.expander("Think Coffee menu catalog (from public menus)"):
    show_dataframe(menu_catalog, hide_index=True)

with st.expander("Data source & tiers"):
    st.write(f"**Sales file:** {data_label}")
    if using_demo_map:
        st.write("Demo mode maps generic items → Think Coffee SKUs via `data/demo_item_map.csv`.")
    st.markdown(
        """
        **Free pilot:** upload your POS CSV, weekly order list, keep using forever.

        **Partnership:** multi-store rollup, custom recipes, waste tracking, supplier lead times, accuracy tuning.
        """
    )
