# StockSight

Cafe inventory forecasting: given POS sales history, recommend how much of each ingredient to order next week.

Built for the Think Coffee problem — *"running out of bacon"* and *"not wanting to overspend"* — using public cafe sales data as a demo until real store data is available.

## Quick start

```bash
pip install -r requirements.txt
python run.py
```

Streamlit UI:

```bash
streamlit run app.py
```

## Data

**Bundled:** `Cleaned_DataSet.csv` — Kaggle cafe sales practice dataset (~9,700 transactions, 2023). Good for demo; not real Think Coffee data.

**Recipes:** `data/recipes.csv` — Think Coffee menu → ingredients (from public menus).

**Demo mapping:** `data/demo_item_map.csv` — maps bundled sales to Think Coffee SKUs.

**Yelp signals:** `data/yelp_reviews_bundled.csv` + `data/yelp_popular_items.csv` — shipped with the repo, no API key needed on deploy.

## Deploy (Streamlit Community Cloud — recommended)

**Yes, this is ready to deploy** for a demo/pilot. Best platform: **[Streamlit Community Cloud](https://streamlit.io/cloud)** (free, built for this exact use case).

1. Push repo to GitHub (include `app.py`, `requirements.txt`, `Cleaned_DataSet.csv`, `data/`, `src/`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select repo, branch, main file: `app.py`
4. Deploy — no secrets needed for the demo (Yelp fetch runs locally, not on deploy)

Share the URL in your Think Coffee email instead of asking them to run code locally.

| Platform | Best for | Notes |
|----------|----------|-------|
| **Streamlit Cloud** | This MVP | Free, zero config, CSV upload works |
| Render / Railway | Production later | If you add auth, DB, scheduled jobs |
| Local laptop | In-person demo | What you use walking into the cafe |

## Think Coffee next step

See `PITCH.md` for email template. Deploy to Streamlit Cloud and link the live URL in your email.
