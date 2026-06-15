"""StockSight — cafe inventory order recommendations from POS sales."""

import argparse
from pathlib import Path

from src.forecast import forecast_next_week
from src.ingest import daily_item_sales, load_sales
from src.menu import apply_demo_menu_labels, load_demo_item_map
from src.recommend import format_order_summary, load_recipes, weekly_order_list

ROOT = Path(__file__).parent
DEFAULT_SALES = ROOT / "Cleaned_DataSet.csv"
DEFAULT_RECIPES = ROOT / "data" / "recipes.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="StockSight weekly order forecast")
    parser.add_argument("--sales", type=Path, default=DEFAULT_SALES)
    parser.add_argument("--recipes", type=Path, default=DEFAULT_RECIPES)
    parser.add_argument("--buffer", type=float, default=0.15, help="Safety buffer (0.15 = 15%%)")
    parser.add_argument("--lead-time", type=int, default=2, help="Supplier lead time in days")
    parser.add_argument(
        "--think-coffee-demo",
        action="store_true",
        default=True,
        help="Map bundled demo items to Think Coffee menu (default: on)",
    )
    parser.add_argument("--no-demo-map", action="store_true", help="Disable Think Coffee demo mapping")
    args = parser.parse_args()

    sales = load_sales(args.sales)
    if args.think_coffee_demo and not args.no_demo_map:
        sales = apply_demo_menu_labels(sales, load_demo_item_map())

    daily = daily_item_sales(sales)
    forecast = forecast_next_week(daily, safety_buffer=args.buffer)
    recipes = load_recipes(args.recipes)
    orders = weekly_order_list(forecast, recipes, lead_time_days=args.lead_time)

    print(f"Sales: {len(sales):,} transactions | {daily['date'].min().date()} to {daily['date'].max().date()}")
    print(f"Menu items: {', '.join(sorted(daily['item'].unique()))}\n")
    print(format_order_summary(orders))


if __name__ == "__main__":
    main()
