import math

import pytest

from supplyiq.eval import forecast_metrics as fm


def test_known_values():
    y, p = [10, 0, 5], [8, 1, 5]
    assert fm.mae(y, p) == pytest.approx(1.0)
    assert fm.rmse(y, p) == pytest.approx(math.sqrt(5 / 3))
    assert fm.wape(y, p) == pytest.approx(3 / 15)


def test_perfect_forecast_is_zero():
    assert fm.evaluate([1, 2, 3], [1, 2, 3]) == {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "n": 3}


def test_wape_is_nan_when_all_actuals_zero():
    assert math.isnan(fm.wape([0, 0], [1, 2]))


def test_bad_inputs_raise():
    with pytest.raises(ValueError, match="shape mismatch"):
        fm.mae([1, 2], [1])
    with pytest.raises(ValueError, match="empty"):
        fm.rmse([], [])
    with pytest.raises(ValueError, match="NaN"):
        fm.wape([1, float("nan")], [1, 2])
