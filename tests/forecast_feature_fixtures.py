"""Synthetic M5-shaped data with the calendar columns the feature code needs (events, SNAP, state)."""
from supplyiq.forecasting import m5_data
from tests.forecast_fixtures import HORIZON, N_DAYS, STORES, synthetic_m5


def synthetic_long_with_calendar():
    sales, cal, prices = synthetic_m5()
    sales["state_id"] = "CA"
    cal = cal.copy()
    cal["event_name_1"] = None
    cal["event_type_1"] = None
    cal.loc[cal["d_num"] == 20, "event_name_1"] = "TestEvent"
    cal.loc[cal["d_num"] == 20, "event_type_1"] = "Sporting"
    cal["snap_CA"] = (cal["d_num"] % 3 == 0).astype(int)
    cal["snap_TX"] = 0
    cal["snap_WI"] = 0
    splits = m5_data.compute_splits(N_DAYS, HORIZON)
    long = m5_data.to_long(sales, cal, prices, ["F_1", "F_2", "H_1", "H_2"], STORES)
    return m5_data.assign_split(long, splits), splits
