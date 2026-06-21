"""Think Coffee menu labels and demo data mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MAP = ROOT / "data" / "demo_item_map.csv"
DEFAULT_CATALOG = ROOT / "data" / "menu_catalog.csv"
DEFAULT_YELP = ROOT / "data" / "yelp_popular_items.csv"


def load_yelp_reviews() -> pd.DataFrame:
    """Bundled reviews — no local script or API key required on deploy."""
    from src.yelp_data import load_bundled_reviews

    return load_bundled_reviews()


def load_demo_item_map(csv_path: str | Path = DEFAULT_MAP) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["demo_item"] = df["demo_item"].str.strip().str.lower()
    df["menu_item"] = df["menu_item"].str.strip().str.lower()
    return df


def load_menu_catalog(csv_path: str | Path = DEFAULT_CATALOG) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_yelp_popular(csv_path: str | Path = DEFAULT_YELP) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def apply_demo_menu_labels(sales: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Map generic demo POS items to Think Coffee menu SKUs for recipes."""
    out = sales.copy()
    mapping = dict(zip(item_map["demo_item"], item_map["menu_item"]))
    labels = dict(zip(item_map["demo_item"], item_map["menu_label"]))
    out["item"] = out["item"].map(mapping).fillna(out["item"])
    out["menu_label"] = out["item"].map(
        {v: labels.get(k, v) for k, v in mapping.items()}
    ).fillna(out["item"])
    return out


def item_display_names(daily: pd.DataFrame, item_map: pd.DataFrame) -> dict[str, str]:
    """menu_item key -> human label for charts and tables."""
    labels = dict(zip(item_map["menu_item"], item_map["menu_label"]))
    demo_labels = dict(zip(item_map["demo_item"], item_map["menu_label"]))
    names = {**demo_labels, **labels}
    return {item: names.get(item, item.replace("_", " ").title()) for item in daily["item"].unique()}
