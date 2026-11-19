"""M5 loading, reproducible subset selection and chronological splits.

Leakage rules baked in:
  * the subset (which items are kept) is chosen using the TRAIN window only,
  * splits are chronological blocks of whole days (train < val < test).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def day_columns(columns) -> list[str]:
    """Return the d_1, d_2, ... columns in numeric order."""
    cols = [c for c in columns if str(c).startswith("d_")]
    return sorted(cols, key=lambda c: int(c[2:]))


def load_sales_wide(path) -> pd.DataFrame:
    """Read an M5 sales file (wide format) with compact integer dtypes."""
    header = pd.read_csv(path, nrows=0).columns
    dtypes = {c: "int16" for c in day_columns(header)}
    return pd.read_csv(path, dtype=dtypes)


def load_calendar(path) -> pd.DataFrame:
    cal = pd.read_csv(path, parse_dates=["date"])
    cal["d_num"] = cal["d"].str[2:].astype(int)
    return cal


def load_prices(path) -> pd.DataFrame:
    return pd.read_csv(path)


def compute_splits(n_days: int, horizon: int = 28) -> dict[str, tuple[int, int]]:
    """Chronological day-index ranges (inclusive): test = last `horizon` days, val = the block before."""
    if n_days <= 2 * horizon:
        raise ValueError(f"need more than {2 * horizon} days, got {n_days}")
    return {
        "train": (1, n_days - 2 * horizon),
        "val": (n_days - 2 * horizon + 1, n_days - horizon),
        "test": (n_days - horizon + 1, n_days),
    }


def select_subset(
    sales_wide: pd.DataFrame,
    stores: list[str],
    categories: list[str],
    items_per_category: int,
    train_end_day: int,
) -> list[str]:
    """Top-N items per category by total units sold in `stores` during days 1..train_end_day.

    Only the train window is used, so no information from val/test decides what we model.
    Ties are broken by item_id, so the result is deterministic.
    """
    train_cols = [c for c in day_columns(sales_wide.columns) if int(c[2:]) <= train_end_day]
    sub = sales_wide[sales_wide["store_id"].isin(stores) & sales_wide["cat_id"].isin(categories)]
    totals = sub[train_cols].astype("int64").sum(axis=1)
    per_item = sub.assign(total=totals).groupby(["cat_id", "item_id"], as_index=False)["total"].sum()
    chosen: list[str] = []
    for cat in categories:
        top = (
            per_item[per_item["cat_id"] == cat]
            .sort_values(["total", "item_id"], ascending=[False, True])
            .head(items_per_category)
        )
        chosen.extend(top["item_id"].tolist())
    return chosen


def resolve_items(
    sales_wide: pd.DataFrame,
    stores: list[str],
    categories: list[str],
    items_per_category: int,
    train_end_day: int,
    explicit_items: list[str] | None = None,
) -> tuple[list[str], str]:
    """Return (items, how_they_were_chosen).

    If `explicit_items` is given (e.g. the items covered by the inventory dataset) they are used as-is,
    after checking that each exists in every selected store. Otherwise the train-window rule is used.
    """
    if not explicit_items:
        items = select_subset(sales_wide, stores, categories, items_per_category, train_end_day)
        return items, "rule: top items per category by units sold in the TRAIN window only"
    items = list(dict.fromkeys(explicit_items))
    in_stores = sales_wide[sales_wide["store_id"].isin(stores)]
    n_stores = in_stores.groupby("item_id")["store_id"].nunique()
    unknown = [i for i in items if i not in n_stores.index]
    partial = [i for i in items if i in n_stores.index and n_stores[i] < len(stores)]
    if unknown or partial:
        raise ValueError(f"explicit items invalid: not in M5 data={unknown}, missing in some stores={partial}")
    return items, "explicit list from configs/forecast.yaml"


def to_long(
    sales_wide: pd.DataFrame,
    calendar: pd.DataFrame,
    prices: pd.DataFrame,
    items: list[str],
    stores: list[str],
    drop_unpriced: bool = True,
) -> pd.DataFrame:
    """Wide -> long (one row per item, store, day) joined with calendar and prices."""
    sub = sales_wide[sales_wide["item_id"].isin(items) & sales_wide["store_id"].isin(stores)]
    long = sub.melt(
        id_vars=ID_COLS,
        value_vars=day_columns(sales_wide.columns),
        var_name="d",
        value_name="sales",
    )
    long = long.merge(calendar, on="d", how="left")
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    if drop_unpriced:
        long = long.dropna(subset=["sell_price"])
    long = long.sort_values(["item_id", "store_id", "d_num"]).reset_index(drop=True)
    return long


def assign_split(df: pd.DataFrame, splits: dict[str, tuple[int, int]]) -> pd.DataFrame:
    df = df.copy()
    df["split"] = ""
    for name, (lo, hi) in splits.items():
        df.loc[df["d_num"].between(lo, hi), "split"] = name
    return df


def summarize(long: pd.DataFrame) -> dict:
    """Quick facts to sanity-check the subset (M5 is intermittent: many zero-sales days)."""
    return {
        "rows": len(long),
        "series": int(long.groupby(["item_id", "store_id"]).ngroups),
        "date_min": str(long["date"].min().date()),
        "date_max": str(long["date"].max().date()),
        "rows_per_split": {k: int(v) for k, v in long["split"].value_counts().items()},
        "zero_sales_share_by_category": {
            k: round(float(v), 3) for k, v in long.groupby("cat_id")["sales"].apply(lambda s: (s == 0).mean()).items()
        },
        "mean_daily_sales_by_category": {
            k: round(float(v), 3) for k, v in long.groupby("cat_id")["sales"].mean().items()
        },
    }


def load_table(stem: Path) -> pd.DataFrame:
    """Read the table written by save_table (parquet or csv.gz)."""
    parquet = stem.with_suffix(".parquet")
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(stem.with_suffix(".csv.gz"), parse_dates=["date"])


def save_table(df: pd.DataFrame, stem: Path) -> Path:
    """Write parquet if pyarrow is installed, otherwise compressed CSV."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        path = stem.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except ImportError:
        path = stem.with_suffix(".csv.gz")
        df.to_csv(path, index=False)
    return path
