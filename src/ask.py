"""Rule-based Q&A over forecast and order outputs — no LLM required."""

from __future__ import annotations

import re

import pandas as pd

# Plain-language aliases → ingredient column in orders
INGREDIENT_ALIASES: dict[str, str] = {
    "bacon": "bacon",
    "egg": "cage_free_egg",
    "eggs": "cage_free_egg",
    "matcha": "matcha",
    "milk": "milk",
    "bread": "brioche",
    "brioche": "brioche",
    "espresso": "espresso",
    "coffee": "coffee_beans",
    "beans": "coffee_beans",
    "lettuce": "lettuce",
    "cheese": "cheddar",
    "cheddar": "cheddar",
    "avocado": "avocado",
    "tortilla": "flour_tortilla",
    "potato": "roasted_potatoes",
    "potatoes": "roasted_potatoes",
}

HELP_TEXT = """Ask about **orders** or **forecasts**, for example:
- How much bacon should we order this week?
- How much matcha do we need?
- How many Spanish lattes next week?

**Note:** StockSight recommends **orders**, not what's currently in the walk-in. On-hand inventory is a future feature."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _find_ingredient(question: str, orders: pd.DataFrame) -> str | None:
    q = _normalize(question)
    known = set(orders["ingredient"].str.lower())
    for alias, ingredient in INGREDIENT_ALIASES.items():
        if alias in q and ingredient in known:
            return ingredient
    for ing in known:
        if ing.replace("_", " ") in q or ing in q:
            return ing
    return None


def _find_menu_item(question: str, forecast: pd.DataFrame, display_names: dict[str, str]) -> str | None:
    q = _normalize(question)
    # Match display labels first (e.g. "spanish latte")
    for item, label in display_names.items():
        if label.lower() in q:
            return item
    for item in forecast["item"].unique():
        readable = item.replace("_", " ")
        if readable in q or item in q:
            return item
    return None


def answer_question(
    question: str,
    orders: pd.DataFrame,
    forecast: pd.DataFrame,
    display_names: dict[str, str],
) -> str:
    """Return a plain-text answer from computed orders and forecast."""
    q = _normalize(question)
    if not q:
        return "Ask a question about an ingredient or menu item."

    if any(w in q for w in ("help", "what can", "how do i", "examples")):
        return HELP_TEXT

    # Inventory on hand — we don't have it
    if any(w in q for w in ("have", "on hand", "in stock", "left", "remaining")) and "order" not in q:
        ing = _find_ingredient(q, orders)
        if ing:
            row = orders[orders["ingredient"] == ing].iloc[0]
            return (
                f"We don't track on-hand inventory yet — only **recommended orders**.\n\n"
                f"For **{ing.replace('_', ' ')}** this week: order **{row['recommended_order']} {row['unit']}** "
                f"(estimated need ~{row['weekly_need']} {row['unit']}, {int(row['lead_time_days'])}-day lead time)."
            )
        return (
            "We don't track what's currently in the walk-in — only **what to order** based on sales forecast. "
            "Try: *How much bacon should we order this week?*"
        )

    ing = _find_ingredient(q, orders)
    if ing:
        row = orders[orders["ingredient"] == ing].iloc[0]
        return (
            f"**{ing.replace('_', ' ').title()}** — order **{row['recommended_order']} {row['unit']}** this week.\n\n"
            f"Estimated need: ~{row['weekly_need']} {row['unit']}. "
            f"Includes safety buffer from your sidebar settings."
        )

    item = _find_menu_item(q, forecast, display_names)
    if item:
        subset = forecast[forecast["item"] == item]
        total = subset["forecast_units"].sum()
        label = display_names.get(item, item.replace("_", " ").title())
        by_day = subset.groupby("dow")["forecast_units"].sum()
        day_lines = ", ".join(f"{d}: {v:.0f}" for d, v in by_day.items())
        return (
            f"**{label}** — forecast **{total:.0f} units** over the next 7 days.\n\n"
            f"By day: {day_lines}."
        )

    if "stockout" in q or "run out" in q:
        bacon = orders[orders["ingredient"] == "bacon"]
        if not bacon.empty:
            r = bacon.iloc[0]
            return (
                f"To reduce bacon stockouts, this week's recommendation is **{r['recommended_order']} slices**. "
                f"Increase the **safety buffer** slider if you're still running out."
            )

    return (
        "I couldn't match that to an ingredient or menu item in the current forecast.\n\n"
        + HELP_TEXT
    )
