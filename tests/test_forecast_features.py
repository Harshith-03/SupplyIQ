import pandas as pd

from supplyiq.forecasting.features import FEATURES, HISTORY_FEATURES, build_features
from tests.forecast_feature_fixtures import synthetic_long_with_calendar
from tests.forecast_fixtures import N_DAYS


def _row(df, item, store, day):
    return df[(df["item_id"] == item) & (df["store_id"] == store) & (df["d_num"] == day)].iloc[0]


def test_all_feature_columns_exist_and_row_count_is_unchanged():
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    assert set(FEATURES) <= set(df.columns)
    assert len(df) == len(long)


def test_lag_28_is_sales_from_28_days_earlier():
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    assert _row(df, "H_1", "S1", 50)["lag_28"] == _row(df, "H_1", "S1", 22)["sales"]
    assert _row(df, "H_1", "S1", 63)["lag_35"] == _row(df, "H_1", "S1", 28)["sales"]


def test_days_missing_from_the_table_give_nan_not_a_shifted_value():
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    # F_1 goes on sale on day 15, so day 14 has no row
    assert pd.isna(_row(df, "F_1", "S1", 42)["lag_28"])
    assert _row(df, "F_1", "S1", 43)["lag_28"] == _row(df, "F_1", "S1", 15)["sales"]


def test_features_never_use_sales_newer_than_28_days_before_the_target():
    long, _ = synthetic_long_with_calendar()
    base = build_features(long, N_DAYS)
    tampered_long = long.copy()
    tampered_long.loc[tampered_long["d_num"] >= 40, "sales"] = 9999
    tampered = build_features(tampered_long, N_DAYS)

    safe = base["d_num"] < 40 + 28                      # their history windows end before day 40
    pd.testing.assert_frame_equal(base.loc[safe, HISTORY_FEATURES], tampered.loc[safe, HISTORY_FEATURES])
    # negative control: later days DO see the change, so the check above is not vacuous
    late = base["d_num"] == N_DAYS
    assert not base.loc[late, HISTORY_FEATURES].equals(tampered.loc[late, HISTORY_FEATURES])


def test_calendar_and_static_features():
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    assert (df["snap"] == (df["d_num"] % 3 == 0).astype(int)).all()      # CA rows follow snap_CA
    assert _row(df, "H_1", "S1", 20)["has_event"] == 1
    assert _row(df, "H_1", "S1", 20)["event_type_code"] == 1
    assert _row(df, "H_1", "S1", 21)["has_event"] == 0
    assert set(df.loc[df["cat_id"] == "FOODS", "cat_code"]) == {0}
    assert set(df.loc[df["cat_id"] == "HOBBIES", "cat_code"]) == {1}
