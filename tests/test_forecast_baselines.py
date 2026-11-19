import numpy as np
import pytest

from supplyiq.forecasting import baselines
from tests.forecast_fixtures import synthetic_long


def test_naive_repeats_last_value():
    assert baselines.naive([1, 2, 7], 3).tolist() == [7, 7, 7]


def test_seasonal_naive_repeats_last_week():
    hist = list(range(1, 15))                       # 1..14, last week = 8..14
    out = baselines.seasonal_naive(hist, 9).tolist()
    assert out == [8, 9, 10, 11, 12, 13, 14, 8, 9]


def test_seasonal_naive_falls_back_when_history_is_short():
    assert baselines.seasonal_naive([3, 4], 4).tolist() == [4, 4, 4, 4]


def test_moving_average():
    assert baselines.moving_average([1, 2, 3, 4], 2, window=2).tolist() == [3.5, 3.5]


def test_baseline_predictions_cover_the_split_with_aligned_days():
    long, splits = synthetic_long()
    pred = baselines.baseline_predictions(long, splits, "val", "naive")
    lo, hi = splits["val"]
    assert pred["d_num"].min() == lo and pred["d_num"].max() == hi
    assert pred["y_pred"].notna().all()
    # F_1 only goes on sale on day 15 (the first val day), so it has no history and is skipped
    assert "F_1" not in set(pred["item_id"])
    assert pred.groupby(["item_id", "store_id"]).ngroups == 6


def test_baseline_uses_only_history_before_the_split():
    long, splits = synthetic_long()
    before = baselines.baseline_predictions(long, splits, "val", "seasonal_naive_7")
    tampered = long.copy()
    tampered.loc[tampered["d_num"] >= splits["val"][0], "sales"] = 9999     # change val/test actuals
    after = baselines.baseline_predictions(tampered, splits, "val", "seasonal_naive_7")
    assert np.array_equal(before["y_pred"].to_numpy(), after["y_pred"].to_numpy())


def test_unknown_method_raises():
    long, splits = synthetic_long()
    with pytest.raises(KeyError):
        baselines.baseline_predictions(long, splits, "val", "nope")
