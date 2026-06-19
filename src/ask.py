"""Rule-based Q&A over forecast and order outputs — no LLM required."""

from __future__ import annotations

import re

import pandas as pd

from src.forecast import DOW_NAMES

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
    "fruit": "fruit",
    "yogurt": "yogurt",
    "flour": "flour",
    "butter": "butter",
    "chocolate": "chocolate_sauce",
    "tea": "tea_leaves",
    "chai": "chai_concentrate",
    "salmon": "smoked_salmon",
    "ham": "ham",
    "chicken": "roasted_chicken",
    "focaccia": "focaccia",
    "peanut butter": "peanut_butter",
    "pb": "peanut_butter",
    "jam": "strawberry_preserves",
}

MENU_ALIASES: dict[str, str] = {
    "spanish latte": "spanish_latte",
    "chai latte": "chai_latte",
    "cold brew": "cold_brew",
    "matcha lemonade": "strawberry_matcha_lemonade",
    "strawberry matcha": "strawberry_matcha_lemonade",
    "almond croissant": "almond_croissant",
    "egg sandwich": "egg_and_cheese",
    "egg and cheese": "egg_and_cheese",
    "sandwich": "egg_and_cheese",
    "sandwiches": "egg_and_cheese",
    "vegan sandwich": "vegan_sandwich",
    "graham cracker": "graham_cracker_latte",
    "chocolate cookie": "chocolate_cookie",
    "cookie": "chocolate_cookie",
}

EXAMPLE_QUESTIONS: list[str] = [
    "How much bacon should we order this week?",
    "How much matcha do we need?",
    "How many Spanish lattes next week?",
    "What's our busiest day for sandwiches?",
    "Which ingredients should we order the most?",
    "Will we run out of bacon?",
    "How much milk should we order?",
    "Forecast almond croissants for Saturday",
    "What do we order for egg sandwiches?",
    "How does the safety buffer affect orders?",
    "Summarize this week's order list",
    "How many chai lattes do we expect?",
]

HELP_TEXT = """Ask about **orders**, **forecasts**, or **patterns**. Examples:

**Ingredients:** *How much bacon / matcha / milk should we order?*  
**Menu items:** *How many Spanish lattes next week?*  
**Operations:** *Busiest day? Top ingredients? Will we run out of bacon?*  
**Settings:** *How does the safety buffer work?*

**Note:** StockSight recommends **orders**, not what's in the walk-in today. On-hand inventory is a future feature."""


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _fmt_ingredient(name: str) -> str:
    return name.replace("_", " ").title()


def _find_ingredient(question: str, orders: pd.DataFrame) -> str | None:
    q = _normalize(question)
    known = set(orders["ingredient"].str.lower())
    for alias in sorted(INGREDIENT_ALIASES, key=len, reverse=True):
        ingredient = INGREDIENT_ALIASES[alias]
        if alias in q and ingredient in known:
            return ingredient
    for ing in sorted(known, key=len, reverse=True):
        if ing.replace("_", " ") in q or ing in q.split():
            return ing
    return None


def _find_menu_item(question: str, forecast: pd.DataFrame, display_names: dict[str, str]) -> str | None:
    q = _normalize(question)
    for phrase in sorted(MENU_ALIASES, key=len, reverse=True):
        if phrase in q:
            item = MENU_ALIASES[phrase]
            if item in forecast["item"].values:
                return item
    for item, label in sorted(display_names.items(), key=lambda x: len(x[1]), reverse=True):
        if label.lower() in q:
            return item
    for item in forecast["item"].unique():
        readable = item.replace("_", " ")
        if readable in q:
            return item
    return None


def _extract_dow(question: str) -> str | None:
    q = _normalize(question)
    for name in DOW_NAMES:
        if name.lower() in q:
            return name
    day_map = {
        "saturday": "Sat",
        "sunday": "Sun",
        "monday": "Mon",
        "tuesday": "Tue",
        "wednesday": "Wed",
        "thursday": "Thu",
        "friday": "Fri",
    }
    for key, val in day_map.items():
        if key in q:
            return val
    return None


def _classify_question(
    question: str,
    orders: pd.DataFrame,
    forecast: pd.DataFrame,
    display_names: dict[str, str],
) -> dict:
    """Detect intent and entities for contextual follow-ups."""
    q = _normalize(question)
    ingredient = _find_ingredient(q, orders)
    menu_item = _find_menu_item(q, forecast, display_names)
    dow = _extract_dow(q)
    label = display_names.get(menu_item, menu_item.replace("_", " ").title()) if menu_item else None

    if any(w in q for w in ("help", "what can", "how do i", "example")):
        intent = "help"
    elif any(w in q for w in ("summarize", "summary", "overview")) and "order" in q:
        intent = "summarize"
    elif any(w in q for w in ("top", "most", "biggest", "largest")) and any(
        w in q for w in ("ingredient", "order", "stock", "buy")
    ):
        intent = "top_orders"
    elif "buffer" in q or "safety" in q:
        intent = "buffer"
    elif any(w in q for w in ("busiest", "busy day", "peak day")):
        intent = "busiest_day"
    elif any(w in q for w in ("stockout", "run out", "running out", "will we run")):
        intent = "stockout"
    elif q.startswith("why") or "where does" in q or "what uses" in q:
        intent = "why"
    elif any(w in q for w in ("last week", "historically", "sold last")):
        intent = "history"
    elif any(w in q for w in ("have", "on hand", "in stock", "left")) and "order" not in q:
        intent = "on_hand"
    elif menu_item and dow:
        intent = "menu_day"
    elif menu_item:
        intent = "menu_forecast"
    elif ingredient:
        intent = "ingredient_order"
    else:
        intent = "unknown"

    return {
        "intent": intent,
        "ingredient": ingredient,
        "menu_item": menu_item,
        "menu_label": label,
        "dow": dow,
    }


def _related_ingredients_for_menu(menu_item: str, recipes: pd.DataFrame | None) -> list[str]:
    if recipes is None or recipes.empty:
        return []
    ings = recipes.loc[recipes["menu_item"] == menu_item, "ingredient"].tolist()
    return [_fmt_ingredient(i) for i in ings[:3]]


def _dedupe_followups(question: str, suggestions: list[str], limit: int = 3) -> list[str]:
    q_norm = _normalize(question)
    seen: set[str] = set()
    out: list[str] = []
    for s in suggestions:
        if _normalize(s) == q_norm:
            continue
        key = _normalize(s)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _answer_top_orders(orders: pd.DataFrame, n: int = 5) -> str:
    top = orders.nlargest(n, "recommended_order")
    lines = [
        f"{i + 1}. **{_fmt_ingredient(r['ingredient'])}** — {r['recommended_order']} {r['unit']}"
        for i, (_, r) in enumerate(top.iterrows())
    ]
    return "**Top ingredients to order this week:**\n\n" + "\n".join(lines)


def _answer_summarize(orders: pd.DataFrame) -> str:
    top = orders.iloc[0]
    bacon_line = ""
    bacon = orders[orders["ingredient"] == "bacon"]
    if not bacon.empty:
        b = bacon.iloc[0]
        bacon_line = f"\n- **Bacon** (stockout risk): {b['recommended_order']} {b['unit']}"
    return (
        f"**Weekly order summary** — {len(orders)} ingredients tracked.\n\n"
        f"- Largest order: **{_fmt_ingredient(top['ingredient'])}** ({top['recommended_order']} {top['unit']})"
        f"{bacon_line}\n\n"
        f"Ask about any ingredient for details."
    )


def _answer_buffer(safety_buffer: float, orders: pd.DataFrame, ingredient: str | None) -> str:
    pct = int(safety_buffer * 100)
    if ingredient:
        row = orders[orders["ingredient"] == ingredient]
        if row.empty:
            return f"Safety buffer is **{pct}%** — adds extra stock on top of the forecast."
        r = row.iloc[0]
        extra = r["recommended_order"] - r["weekly_need"]
        return (
            f"**Safety buffer: {pct}%**\n\n"
            f"For **{_fmt_ingredient(ingredient)}**: need ~{r['weekly_need']} {r['unit']}, "
            f"order **{r['recommended_order']} {r['unit']}** (~{extra:.1f} extra from buffer).\n\n"
            f"Higher buffer = fewer stockouts, more waste risk."
        )
    return (
        f"**Safety buffer: {pct}%** — inflates forecasts before building orders. "
        f"Raise it if you run out; lower it if you over-order."
    )


def _answer_busiest_day(
    forecast: pd.DataFrame,
    display_names: dict[str, str],
    item: str | None,
) -> str:
    subset = forecast if item is None else forecast[forecast["item"] == item]
    if subset.empty:
        return "No forecast data for that item."
    by_dow = subset.groupby("dow")["forecast_units"].sum().reindex(DOW_NAMES, fill_value=0)
    busiest = by_dow.idxmax()
    label = display_names.get(item, "all menu items") if item else "all menu items"
    lines = ", ".join(f"{d}: {v:.0f}" for d, v in by_dow.items() if v > 0)
    return f"**Busiest day for {label}:** **{busiest}** ({by_dow[busiest]:.0f} units).\n\nFull week: {lines}."


def _answer_why_ingredient(ingredient: str, recipes: pd.DataFrame | None) -> str:
    if recipes is None or recipes.empty:
        return ""
    sources = recipes[recipes["ingredient"] == ingredient]["menu_item"].unique()
    if len(sources) == 0:
        return ""
    names = ", ".join(s.replace("_", " ") for s in sources[:5])
    extra = f" (+{len(sources) - 5} more)" if len(sources) > 5 else ""
    return f"\n\n**Comes from menu items:** {names}{extra}."


def answer_question(
    question: str,
    orders: pd.DataFrame,
    forecast: pd.DataFrame,
    display_names: dict[str, str],
    *,
    daily: pd.DataFrame | None = None,
    safety_buffer: float = 0.15,
    recipes: pd.DataFrame | None = None,
) -> str:
    q = _normalize(question)
    if not q:
        return "Ask a question about an ingredient or menu item."

    if any(w in q for w in ("help", "what can", "how do i", "example")):
        return HELP_TEXT

    if any(w in q for w in ("summarize", "summary", "overview")) and "order" in q:
        return _answer_summarize(orders)

    if any(w in q for w in ("top", "most", "biggest", "largest")) and any(
        w in q for w in ("ingredient", "order", "stock", "buy")
    ):
        return _answer_top_orders(orders)

    if "buffer" in q or "safety" in q:
        return _answer_buffer(safety_buffer, orders, _find_ingredient(q, orders))

    if any(w in q for w in ("busiest", "busy day", "peak day")):
        item = _find_menu_item(q, forecast, display_names)
        return _answer_busiest_day(forecast, display_names, item)

    if any(w in q for w in ("stockout", "run out", "running out", "will we run")):
        ing = _find_ingredient(q, orders) or "bacon"
        row = orders[orders["ingredient"] == ing]
        if row.empty:
            return _answer_top_orders(orders, 3)
        r = row.iloc[0]
        return (
            f"Order **{r['recommended_order']} {r['unit']}** of {_fmt_ingredient(ing)}. "
            f"Still running out? Raise **safety buffer** (now {int(safety_buffer * 100)}%)."
            + _answer_why_ingredient(ing, recipes)
        )

    if q.startswith("why") or "where does" in q or "what uses" in q:
        ing = _find_ingredient(q, orders)
        if ing:
            row = orders[orders["ingredient"] == ing].iloc[0]
            return (
                f"**{_fmt_ingredient(ing)}** — order **{row['recommended_order']} {row['unit']}** this week."
                + _answer_why_ingredient(ing, recipes)
            )

    if daily is not None and any(w in q for w in ("last week", "historically", "sold last")):
        item = _find_menu_item(q, forecast, display_names)
        if item:
            total = daily[daily["item"] == item].tail(7)["units_sold"].sum()
            label = display_names.get(item, item)
            return f"**{label}** — last 7 days sold: **{total:.0f} units** (historical)."

    if any(w in q for w in ("have", "on hand", "in stock", "left")) and "order" not in q:
        ing = _find_ingredient(q, orders)
        if ing:
            row = orders[orders["ingredient"] == ing].iloc[0]
            return (
                f"No on-hand tracking yet — only **recommended orders**.\n\n"
                f"**{_fmt_ingredient(ing)}**: order **{row['recommended_order']} {row['unit']}** this week."
            )
        return "We track **orders**, not walk-in inventory. Try: *How much bacon should we order?*"

    ing = _find_ingredient(q, orders)
    if ing:
        row = orders[orders["ingredient"] == ing].iloc[0]
        return (
            f"**{_fmt_ingredient(ing)}** — order **{row['recommended_order']} {row['unit']}** this week.\n\n"
            f"Need ~{row['weekly_need']} {row['unit']} (+ {int(safety_buffer * 100)}% buffer)."
            + _answer_why_ingredient(ing, recipes)
        )

    item = _find_menu_item(q, forecast, display_names)
    if item:
        subset = forecast[forecast["item"] == item]
        label = display_names.get(item, item.replace("_", " ").title())
        dow = _extract_dow(q)
        if dow:
            units = subset.loc[subset["dow"] == dow, "forecast_units"].sum()
            return f"**{label}** on **{dow}**: **{units:.0f} units** forecast."
        total = subset["forecast_units"].sum()
        by_day = subset.groupby("dow")["forecast_units"].sum()
        lines = ", ".join(f"{d}: {v:.0f}" for d, v in by_day.items())
        return f"**{label}** — **{total:.0f} units** next 7 days.\n\nBy day: {lines}."

    suggestions = orders["ingredient"].head(6).apply(_fmt_ingredient).tolist()
    return "Try: " + ", ".join(suggestions) + ".\n\n" + HELP_TEXT


def follow_up_suggestions(
    question: str,
    orders: pd.DataFrame,
    forecast: pd.DataFrame,
    display_names: dict[str, str],
    recipes: pd.DataFrame | None = None,
) -> list[str]:
    """Contextual follow-ups based on what the user just asked."""
    ctx = _classify_question(question, orders, forecast, display_names)
    intent = ctx["intent"]
    ing = ctx["ingredient"]
    label = ctx["menu_label"]
    dow = ctx["dow"]
    ing_name = _fmt_ingredient(ing) if ing else None

    suggestions: list[str] = []

    if intent == "ingredient_order" and ing:
        suggestions = [
            f"Why do we need so much {ing_name}?",
            f"Will we run out of {ing_name}?",
            f"How does the safety buffer affect {ing_name}?",
        ]
    elif intent == "stockout" and ing:
        suggestions = [
            f"Why do we need so much {ing_name}?",
            f"How much {ing_name} should we order this week?",
            "How does the safety buffer affect orders?",
        ]
    elif intent == "why" and ing:
        suggestions = [
            f"Will we run out of {ing_name}?",
            f"How much {ing_name} should we order this week?",
            f"How does the safety buffer affect {ing_name}?",
        ]
    elif intent == "on_hand" and ing:
        suggestions = [
            f"How much {ing_name} should we order this week?",
            f"Why do we need so much {ing_name}?",
            f"Will we run out of {ing_name}?",
        ]
    elif intent == "buffer":
        if ing:
            suggestions = [
                f"How much {ing_name} should we order this week?",
                f"Will we run out of {ing_name}?",
                "Summarize this week's order list",
            ]
        else:
            suggestions = [
                "How much bacon should we order this week?",
                "Which ingredients should we order the most?",
                "Summarize this week's order list",
            ]
    elif intent == "summarize":
        top = orders.iloc[0]
        top_name = _fmt_ingredient(top["ingredient"])
        suggestions = [
            f"How much {top_name} should we order this week?",
            "Which ingredients should we order the most?",
            "How does the safety buffer affect orders?",
        ]
    elif intent == "top_orders":
        if ing:
            suggestions = [
                f"Why do we need so much {ing_name}?",
                "Summarize this week's order list",
                f"Will we run out of {ing_name}?",
            ]
        else:
            top = orders.iloc[0]
            top_name = _fmt_ingredient(top["ingredient"])
            suggestions = [
                f"How much {top_name} should we order this week?",
                "Summarize this week's order list",
                "How does the safety buffer affect orders?",
            ]
    elif intent == "busiest_day":
        if label:
            related = _related_ingredients_for_menu(ctx["menu_item"], recipes)
            suggestions = [
                f"How many {label}s next week?",
                f"Forecast {label}s for Saturday",
            ]
            if related:
                suggestions.append(f"How much {related[0]} should we order this week?")
            else:
                suggestions.append("How does the safety buffer affect orders?")
        else:
            suggestions = [
                "How many Spanish lattes next week?",
                "What's our busiest day for sandwiches?",
                "Forecast almond croissants for Saturday",
            ]
    elif intent == "menu_day" and label and dow:
        suggestions = [
            f"How many {label}s next week?",
            f"What's our busiest day for {label.lower()}s?",
            f"Forecast {label}s for Saturday" if dow != "Sat" else f"Forecast {label}s for Sunday",
        ]
    elif intent == "menu_forecast" and label:
        related = _related_ingredients_for_menu(ctx["menu_item"], recipes)
        suggestions = [
            f"Forecast {label}s for Saturday",
            f"What's our busiest day for {label.lower()}s?",
        ]
        if related:
            suggestions.append(f"How much {related[0]} should we order this week?")
        else:
            suggestions.append("Which ingredients should we order the most?")
    elif intent == "history" and label:
        suggestions = [
            f"How many {label}s next week?",
            f"Forecast {label}s for Saturday",
            f"What's our busiest day for {label.lower()}s?",
        ]
    elif intent == "help":
        suggestions = EXAMPLE_QUESTIONS[:3]
    else:
        if ing:
            suggestions = [
                f"Why do we need so much {ing_name}?",
                f"Will we run out of {ing_name}?",
                "Summarize this week's order list",
            ]
        elif label:
            suggestions = [
                f"How many {label}s next week?",
                f"Forecast {label}s for Saturday",
                "Which ingredients should we order the most?",
            ]
        else:
            suggestions = [
                "How much bacon should we order this week?",
                "Which ingredients should we order the most?",
                "Summarize this week's order list",
            ]

    return _dedupe_followups(question, suggestions)
