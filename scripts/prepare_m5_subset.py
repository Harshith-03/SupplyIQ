"""Step 1 of the XGBoost pipeline: build the M5 subset and chronological splits.

Usage (from repo root, with the M5 CSVs in data/raw/m5/):
    python scripts/prepare_m5_subset.py

Writes:
    data/reference/m5_subset.json    small, COMMITTED: the agreed items/stores/splits (shared with the whole team)
    data/processed/m5_subset_long.*  large, NOT committed: model-ready long table
"""
import json
from pathlib import Path

import yaml

from supplyiq.forecasting import m5_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "configs/forecast.yaml").read_text())
    raw = ROOT / cfg["data"]["raw_dir"]
    sales = m5_data.load_sales_wide(raw / cfg["data"]["sales_file"])
    calendar = m5_data.load_calendar(raw / cfg["data"]["calendar_file"])
    prices = m5_data.load_prices(raw / cfg["data"]["prices_file"])

    n_days = len(m5_data.day_columns(sales.columns))
    splits = m5_data.compute_splits(n_days, cfg["splits"]["horizon_days"])
    sub = cfg["subset"]
    items, source = m5_data.resolve_items(
        sales,
        sub["stores"],
        sub["categories"],
        sub["items_per_category"],
        splits["train"][1],
        sub.get("items"),
    )
    long = m5_data.to_long(sales, calendar, prices, items, sub["stores"], cfg["drop_unpriced_rows"])
    long = m5_data.assign_split(long, splits)
    summary = m5_data.summarize(long)

    out_ref = ROOT / "data/reference/m5_subset.json"
    out_ref.parent.mkdir(parents=True, exist_ok=True)
    out_ref.write_text(
        json.dumps(
            {
                "source": "M5 Forecasting - Accuracy (Kaggle), sales_train_evaluation.csv",
                "stores": sub["stores"],
                "categories": sub["categories"],
                "items": items,
                "selection_source": source,
                "n_days": n_days,
                "splits_day_index_inclusive": {k: list(v) for k, v in splits.items()},
                "summary": summary,
            },
            indent=2,
        )
        + "\n"
    )
    path = m5_data.save_table(long, ROOT / cfg["data"]["processed_dir"] / "m5_subset_long")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_ref.relative_to(ROOT)} and {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
