"""Bundled + optional live Yelp review data. No local scripts required for deploy."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BUNDLED_REVIEWS = ROOT / "data" / "yelp_reviews_bundled.csv"
BUNDLED_POPULAR = ROOT / "data" / "yelp_popular_items.csv"
BUNDLED_LOCATIONS = ROOT / "data" / "yelp_locations_bundled.csv"

API_BASE = "https://api.yelp.com/v3"
MAX_API_REVIEWS = 3
SEARCH_TERM = "Think Coffee"
SEARCH_LOCATION = "New York, NY"

MENU_KEYWORDS = [
    "spanish latte", "matcha", "cold brew", "chai", "croissant", "almond",
    "bacon", "sandwich", "burrito", "bagel", "muffin", "scone", "granola",
    "pb&j", "peanut butter", "egg", "latte", "espresso", "pastry", "coffee",
]


def load_bundled_reviews() -> pd.DataFrame:
    """Reviews shipped with the repo — works on Streamlit Cloud with zero setup."""
    if not BUNDLED_REVIEWS.exists():
        return pd.DataFrame()
    df = pd.read_csv(BUNDLED_REVIEWS)
    if "store_address" not in df.columns:
        df["store_address"] = "123 4th Ave, NYC"
    if "store_name" not in df.columns:
        df["store_name"] = "Think Coffee"
    return df


def load_bundled_popular() -> pd.DataFrame:
    if not BUNDLED_POPULAR.exists():
        return pd.DataFrame()
    return pd.read_csv(BUNDLED_POPULAR)


def load_bundled_locations() -> pd.DataFrame:
    """All 11 Think Coffee NYC locations (from public store listing)."""
    if not BUNDLED_LOCATIONS.exists():
        return pd.DataFrame()
    return pd.read_csv(BUNDLED_LOCATIONS)


def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}"}


def find_all_locations(api_key: str) -> list[dict]:
    locations: list[dict] = []
    offset = 0
    while offset < 240:
        resp = requests.get(
            f"{API_BASE}/businesses/search",
            headers=_headers(api_key),
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
                    "store_address": loc.get("address1", ""),
                }
            )
        offset += len(businesses)
        if offset >= data.get("total", 0):
            break
    seen: set[str] = set()
    unique: list[dict] = []
    for loc in locations:
        if loc["business_id"] not in seen:
            seen.add(loc["business_id"])
            unique.append(loc)
    return unique


def fetch_reviews_for_location(api_key: str, store: dict) -> list[dict]:
    resp = requests.get(
        f"{API_BASE}/businesses/{store['business_id']}/reviews",
        headers=_headers(api_key),
        params={"limit": 50, "sort_by": "newest"},
        timeout=30,
    )
    resp.raise_for_status()
    rows = []
    for r in resp.json().get("reviews", [])[:MAX_API_REVIEWS]:
        text = r.get("text", "")
        mentions = [k for k in MENU_KEYWORDS if k in text.lower()]
        rows.append(
            {
                "store_name": store["store_name"],
                "store_address": store["store_address"],
                "business_id": store["business_id"],
                "rating": r.get("rating"),
                "user": r.get("user", {}).get("name", ""),
                "date": r.get("time_created", ""),
                "text": text,
                "menu_mentions": "|".join(mentions),
                "source": "yelp_api",
            }
        )
    return rows


def fetch_live_reviews(api_key: str) -> pd.DataFrame:
    """Optional: all NYC Think Coffee locations, 3 reviews each. Requires API key."""
    all_rows: list[dict] = []
    for store in find_all_locations(api_key):
        try:
            all_rows.extend(fetch_reviews_for_location(api_key, store))
        except requests.HTTPError:
            continue
        time.sleep(0.2)
    return pd.DataFrame(all_rows)


def get_reviews(api_key: str | None = None) -> pd.DataFrame:
    """
    Bundled reviews always load. If api_key is set (Streamlit secrets), merge live API data.
    """
    bundled = load_bundled_reviews()
    if not api_key:
        return bundled
    try:
        live = fetch_live_reviews(api_key)
        if live.empty:
            return bundled
        combined = pd.concat([bundled, live], ignore_index=True)
        dedupe_cols = [c for c in ["user", "date", "text"] if c in combined.columns]
        if dedupe_cols:
            combined = combined.drop_duplicates(subset=dedupe_cols, keep="first")
        return combined
    except Exception:
        return bundled


def resolve_api_key(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit.strip() or None
    env = os.environ.get("YELP_API_KEY", "").strip()
    return env or None
