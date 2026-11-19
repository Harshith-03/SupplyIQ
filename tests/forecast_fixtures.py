"""Tiny synthetic M5-shaped tables shared by the forecasting tests (CI has no raw M5 data)."""
import numpy as np
import pandas as pd

from supplyiq.forecasting import m5_data

N_DAYS = 70
HORIZON = 28
STORES = ["S1", "S2"]


def synthetic_m5():
    rng = np.random.default_rng(0)
    rows = []
    for cat, prefix in (("FOODS", "F"), ("HOBBIES", "H")):
        for i in range(1, 5):
            for store in STORES:
                base = rng.integers(5, 20, size=N_DAYS)
                if f"{prefix}_{i}" == "F_4":
                    base[: N_DAYS - HORIZON] = 0      # silent in train/val ...
                    base[N_DAYS - HORIZON:] = 500     # ... huge in the test window
                row = {"id": f"{prefix}_{i}_{store}", "item_id": f"{prefix}_{i}", "dept_id": f"{prefix}_D",
                       "cat_id": cat, "store_id": store, "state_id": "ST"}
                row.update({f"d_{d + 1}": int(v) for d, v in enumerate(base)})
                rows.append(row)
    sales = pd.DataFrame(rows)
    for c in m5_data.day_columns(sales.columns):
        sales[c] = sales[c].astype("int16")
    dates = pd.date_range("2011-01-29", periods=N_DAYS)
    cal = pd.DataFrame({"date": dates, "d": [f"d_{i + 1}" for i in range(N_DAYS)],
                        "wm_yr_wk": [11101 + i // 7 for i in range(N_DAYS)]})
    cal["d_num"] = np.arange(1, N_DAYS + 1)
    price_rows = []
    for item in sales["item_id"].unique():
        for store in STORES:
            for wk in sorted(cal["wm_yr_wk"].unique()):
                if item == "F_1" and wk < 11103:      # F_1 not on sale in the first two weeks
                    continue
                price_rows.append({"store_id": store, "item_id": item, "wm_yr_wk": wk, "sell_price": 2.0})
    return sales, cal, pd.DataFrame(price_rows)


def synthetic_long():
    sales, cal, prices = synthetic_m5()
    splits = m5_data.compute_splits(N_DAYS, HORIZON)
    long = m5_data.to_long(sales, cal, prices, ["F_1", "F_2", "H_1", "H_2"], STORES)
    return m5_data.assign_split(long, splits), splits
