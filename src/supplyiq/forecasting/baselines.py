"""Naive baselines every forecasting model must beat. They use only history up to the split's cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive(history, horizon: int) -> np.ndarray:
    """Repeat the last observed value."""
    return np.full(horizon, float(history[-1]))


def seasonal_naive(history, horizon: int, season: int = 7) -> np.ndarray:
    """Repeat the last `season` days cyclically (weekly pattern by default)."""
    if len(history) < season:
        return naive(history, horizon)
    return np.resize(np.asarray(history[-season:], dtype=float), horizon)


def moving_average(history, horizon: int, window: int = 28) -> np.ndarray:
    """Repeat the mean of the last `window` days."""
    return np.full(horizon, float(np.mean(history[-window:])))


METHODS = {
    "naive": naive,
    "seasonal_naive_7": seasonal_naive,
    "moving_average_28": moving_average,
}


def baseline_predictions(
    long: pd.DataFrame,
    splits: dict[str, tuple[int, int]],
    split_name: str,
    method: str,
) -> pd.DataFrame:
    """Forecast every series over `split_name` using only days before the split starts.

    Returns one row per (item, store, day) in the split with columns y_true and y_pred.
    Series with no history before the cutoff are skipped; a model compared against these
    baselines must be scored on exactly the same rows.
    """
    func = METHODS[method]
    lo, hi = splits[split_name]
    cutoff = lo - 1
    horizon = hi - cutoff
    parts = []
    for _key, g in long.groupby(["item_id", "store_id"], sort=True):
        history = g.loc[g["d_num"] <= cutoff, "sales"].to_numpy()
        target = g[(g["d_num"] >= lo) & (g["d_num"] <= hi)]
        if len(history) == 0 or target.empty:
            continue
        preds = func(history, horizon)
        out = target[["item_id", "store_id", "cat_id", "d_num", "sales"]].copy()
        out["y_pred"] = preds[(target["d_num"] - lo).to_numpy()]
        parts.append(out)
    if not parts:
        raise ValueError("no series with both history and target rows")
    return pd.concat(parts, ignore_index=True).rename(columns={"sales": "y_true"})
