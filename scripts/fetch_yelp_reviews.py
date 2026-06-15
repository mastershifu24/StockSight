"""
Fetch Think Coffee Yelp data via the official Yelp Fusion API.

Yelp returns at most 3 review excerpts PER LOCATION. This script searches all
Think Coffee listings in NYC and fetches 3 reviews each (~33 for 11 stores).

Get a key: https://www.yelp.com/developers/v3/manage_app

Usage:
  set YELP_API_KEY=your_key_here
  python scripts/fetch_yelp_reviews.py
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

SEARCH_TERM = "Think Coffee"
SEARCH_LOCATION = "New York, NY"
ROOT = Path(__file__).resolve().parent.parent
OUT_REVIEWS = ROOT / "data" / "yelp_reviews_fetched.csv"
OUT_POPULAR = ROOT / "data" / "yelp_popular_items.csv"
OUT_LOCATIONS = ROOT / "data" / "yelp_locations.csv"
MANUAL_REVIEWS = ROOT / "data" / "yelp_reviews_manual.csv"

API_BASE = "https://api.yelp.com/v3"
MAX_API_REVIEWS = 3  # Yelp Fusion hard limit per location per request

MENU_KEYWORDS = [
    "spanish latte", "matcha", "cold brew", "chai", "croissant", "almond",
    "bacon", "sandwich", "burrito", "bagel", "muffin", "scone", "granola",
    "pb&j", "peanut butter", "egg", "latte", "espresso", "pastry", "coffee",
]


def _headers() -> dict:
    key = os.environ.get("YELP_API_KEY", "").strip()
    if not key:
        print("ERROR: Set YELP_API_KEY environment variable.")
        print("  1. Create app at https://www.yelp.com/developers/v3/manage_app")
        print("  2. set YELP_API_KEY=your_key   (Windows)")
        print("  3. export YELP_API_KEY=your_key  (Mac/Linux)")
        sys.exit(1)
    return {"Authorization": f"Bearer {key}"}


def find_all_think_coffee_locations() -> list[dict]:
    """Search NYC for every Think Coffee listing on Yelp."""
    locations: list[dict] = []
    offset = 0

    while offset < 240:
        resp = requests.get(
            f"{API_BASE}/businesses/search",
            headers=_headers(),
            params={
                "term": SEARCH_TERM,
                "location": SEARCH_LOCATION,
                "limit": 50,
                "offset": offset,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        businesses = data.get("businesses", [])
        if not businesses:
            break

        for b in businesses:
            if "think coffee" not in b.get("name", "").lower():
                continue
            loc = b.get("location", {})
            locations.append(
                {
                    "business_id": b["id"],
                    "store_name": b["name"],
                    "address": loc.get("address1", ""),
                    "city": loc.get("city", ""),
                    "review_count": b.get("review_count", 0),
                    "rating": b.get("rating", 0),
                }
            )

        offset += len(businesses)
        if offset >= data.get("total", 0):
            break

    # Deduplicate by business_id
    seen: set[str] = set()
    unique: list[dict] = []
    for loc in locations:
        if loc["business_id"] not in seen:
            seen.add(loc["business_id"])
            unique.append(loc)

    return unique


def fetch_reviews(business_id: str) -> list[dict]:
    """Up to 3 review excerpts per location (Yelp platform limit)."""
    resp = requests.get(
        f"{API_BASE}/businesses/{business_id}/reviews",
        headers=_headers(),
        params={"limit": 50, "sort_by": "newest"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("reviews", [])[:MAX_API_REVIEWS]


def extract_menu_mentions(text: str) -> list[str]:
    text_lower = text.lower()
    return [k for k in MENU_KEYWORDS if k in text_lower]


def reviews_to_rows(
    reviews: list[dict],
    store: dict,
    source: str = "yelp_api",
) -> list[dict]:
    rows = []
    for r in reviews:
        text = r.get("text", "")
        mentions = extract_menu_mentions(text)
        rows.append(
            {
                "store_name": store["store_name"],
                "store_address": store["address"],
                "business_id": store["business_id"],
                "rating": r.get("rating"),
                "user": r.get("user", {}).get("name", r.get("user", "")),
                "date": r.get("time_created", r.get("date", "")),
                "text": text,
                "menu_mentions": "|".join(mentions) if mentions else "",
                "source": source,
            }
        )
    return rows


def load_manual_reviews() -> pd.DataFrame:
    if not MANUAL_REVIEWS.exists():
        return pd.DataFrame()
    return pd.read_csv(MANUAL_REVIEWS)


def build_popular_from_reviews(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "menu_mentions" not in df.columns:
        return pd.DataFrame()
    mentioned = df[df["menu_mentions"].astype(str).str.len() > 0].copy()
    exploded = mentioned["menu_mentions"].str.replace(", ", "|").str.split("|").explode()
    counts = exploded.value_counts().reset_index()
    counts.columns = ["item_label", "review_mentions"]
    counts["item_label"] = counts["item_label"].str.title()
    counts["category"] = "from_reviews"
    counts["photo_count"] = 0
    counts["notes"] = "Counted from review text (manual + API, all locations)"
    return counts


def main() -> None:
    print(f"Searching Yelp for all '{SEARCH_TERM}' locations in {SEARCH_LOCATION}...\n")
    locations = find_all_think_coffee_locations()
    if not locations:
        raise RuntimeError("No Think Coffee locations found on Yelp.")

    loc_df = pd.DataFrame(locations)
    loc_df.to_csv(OUT_LOCATIONS, index=False)
    print(f"Found {len(locations)} Think Coffee location(s) on Yelp:")
    for loc in locations:
        print(f"  • {loc['address']} — {loc['review_count']} reviews on Yelp ({loc['rating']}★)")
    print(f"\nSaved locations to {OUT_LOCATIONS}")
    print(f"Fetching up to {MAX_API_REVIEWS} review excerpts per location...\n")

    api_rows: list[dict] = []
    for i, loc in enumerate(locations):
        try:
            reviews = fetch_reviews(loc["business_id"])
            api_rows.extend(reviews_to_rows(reviews, loc))
            print(f"  [{i + 1}/{len(locations)}] {loc['address']}: {len(reviews)} reviews")
        except requests.HTTPError as e:
            print(f"  [{i + 1}/{len(locations)}] {loc['address']}: skipped ({e})")
        time.sleep(0.2)  # polite pacing for API rate limits

    manual_df = load_manual_reviews()
    api_df = pd.DataFrame(api_rows)
    combined = pd.concat([manual_df, api_df], ignore_index=True)
    combined.to_csv(OUT_REVIEWS, index=False)

    print(f"\nSaved {len(combined)} total reviews:")
    print(f"  {len(manual_df)} manual + {len(api_df)} API ({len(locations)} locations × up to 3)")
    print(f"  → {OUT_REVIEWS}")

    popular = build_popular_from_reviews(combined)
    if not popular.empty:
        popular.to_csv(OUT_POPULAR, index=False)
        print(f"  → {OUT_POPULAR}")
        print("\nTop menu mentions across all cataloged reviews:")
        for _, row in popular.head(10).iterrows():
            print(f"  {row['item_label']}: {row['review_mentions']}")


if __name__ == "__main__":
    main()
