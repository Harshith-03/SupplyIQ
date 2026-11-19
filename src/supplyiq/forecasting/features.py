"""Leakage-safe features for the shared demand model.

Core idea: every history-based feature for target day t only uses sales up to day t - 28 (lags >= 28 and
rolling windows that END at t - 28). So one model can forecast any horizon 1..28 days after a cutoff
without feeding its own predictions back in. Calendar and price features describe day t itself; they are
known in advance (calendar) or published in advance (M5 `sell_prices`).

History features are computed on a complete day grid per series, so days missing from the table (item
not yet on sale) give NaN instead of silently shifting the lags. XGBoost handles NaN natively.
"""
from __future__ import annotations

import pandas as pd

MIN_LAG = 28
LAGS = (28, 35, 42, 49, 56, 364)
WINDOWS = (7, 14, 28, 56)

CAT_CODES = {"FOODS": 0, "HOBBIES": 1, "HOUSEHOLD": 2}
DEPT_CODES = {
    "FOODS_1": 0, "FOODS_2": 1, "FOODS_3": 2,
    "HOBBIES_1": 3, "HOBBIES_2": 4,
    "HOUSEHOLD_1": 5, "HOUSEHOLD_2": 6,
}
STORE_CODES = {
    s: i for i, s in enumerate(["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"])
}
EVENT_TYPE_CODES = {"Sporting": 1, "Cultural": 2, "National": 3, "Religious": 4}

HISTORY_FEATURES = (
    [f"lag_{lag}" for lag in LAGS]
    + [f"roll_mean_{w}" for w in WINDOWS]
    + [f"roll_std_{w}" for w in WINDOWS]
    + [f"zero_share_{w}" for w in WINDOWS]
    + ["price_ratio_28"]
)
CALENDAR_FEATURES = ["dow", "month_num", "day_of_month", "week_of_year", "has_event", "event_type_code", "snap"]
STATIC_FEATURES = ["sell_price", "cat_code", "dept_code", "store_code"]
FEATURES = HISTORY_FEATURES + CALENDAR_FEATURES + STATIC_FEATURES


def _series_features(g: pd.DataFrame, max_day: int) -> pd.DataFrame:
    days = pd.RangeIndex(1, max_day + 1, name="d_num")
    idx = g["d_num"].to_numpy()
    sales = pd.Series(g["sales"].to_numpy(dtype=float), index=idx).reindex(days)
    price = pd.Series(g["sell_price"].to_numpy(dtype=float), index=idx).reindex(days)

    base = sales.shift(MIN_LAG)                      # value at row t = sales at day t - 28
    cols = {f"lag_{lag}": sales.shift(lag) for lag in LAGS}
    zero = base.eq(0).astype(float).where(base.notna())
    for w in WINDOWS:
        min_p = max(1, w // 2)
        roll = base.rolling(w, min_periods=min_p)
        cols[f"roll_mean_{w}"] = roll.mean()
        cols[f"roll_std_{w}"] = roll.std()
        cols[f"zero_share_{w}"] = zero.rolling(w, min_periods=min_p).mean()
    cols["price_ratio_28"] = price / price.shift(1).rolling(28, min_periods=1).mean()
    return pd.DataFrame(cols, index=days).reset_index()


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["month_num"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["has_event"] = df["event_name_1"].notna().astype(int)
    df["event_type_code"] = df["event_type_1"].map(EVENT_TYPE_CODES).fillna(0).astype(int)
    df["snap"] = 0
    for state in ("CA", "TX", "WI"):
        mask = df["state_id"] == state
        df.loc[mask, "snap"] = df.loc[mask, f"snap_{state}"]
    df["cat_code"] = df["cat_id"].map(CAT_CODES)
    df["dept_code"] = df["dept_id"].map(DEPT_CODES)
    df["store_code"] = df["store_id"].map(STORE_CODES)
    return df


def build_features(long: pd.DataFrame, max_day: int) -> pd.DataFrame:
    """Return `long` plus all columns in FEATURES. `max_day` is the last day index in the data."""
    parts = []
    for (item, store), g in long.groupby(["item_id", "store_id"], sort=True):
        feats = _series_features(g, max_day)
        feats["item_id"] = item
        feats["store_id"] = store
        parts.append(feats)
    history = pd.concat(parts, ignore_index=True)
    df = long.merge(history, on=["item_id", "store_id", "d_num"], how="left")
    return add_calendar_features(df)
