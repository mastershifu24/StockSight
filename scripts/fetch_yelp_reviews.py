"""CLI wrapper — optional local refresh; deploy uses bundled CSV only."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yelp_data import (  # noqa: E402
    BUNDLED_POPULAR,
    BUNDLED_REVIEWS,
    fetch_live_reviews,
    get_reviews,
    load_bundled_reviews,
    resolve_api_key,
)


def build_popular_from_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "menu_mentions" not in df.columns:
        return pd.DataFrame()
    mentioned = df[df["menu_mentions"].astype(str).str.len() > 0].copy()
    exploded = mentioned["menu_mentions"].str.split("|").explode()
    counts = exploded.value_counts().reset_index()
    counts.columns = ["item_label", "review_mentions"]
    counts["item_label"] = counts["item_label"].str.title()
    counts["category"] = "from_reviews"
    counts["photo_count"] = 0
    counts["notes"] = "Counted from bundled + API review text"
    return counts


def main() -> None:
    api_key = resolve_api_key()
    if not api_key:
        print("No YELP_API_KEY — bundled reviews only (same as deployed app).")
        df = load_bundled_reviews()
        print(f"  {len(df)} reviews in {BUNDLED_REVIEWS.name}")
        return

    print("Fetching live Yelp reviews for all Think Coffee NYC locations...")
    df = get_reviews(api_key)
    df.to_csv(BUNDLED_REVIEWS, index=False)
    print(f"Updated {BUNDLED_REVIEWS} with {len(df)} reviews")

    popular = build_popular_from_reviews(df)
    if not popular.empty:
        popular.to_csv(BUNDLED_POPULAR, index=False)
        print(f"Updated {BUNDLED_POPULAR}")


if __name__ == "__main__":
    main()
