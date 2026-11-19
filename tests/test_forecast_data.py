"""Tests for M5 subset selection and splits, using tiny synthetic M5-shaped data."""
import pytest

from supplyiq.forecasting import m5_data
from tests.forecast_fixtures import HORIZON, N_DAYS, STORES, synthetic_m5


def test_compute_splits_matches_m5_layout():
    assert m5_data.compute_splits(1941, 28) == {"train": (1, 1885), "val": (1886, 1913), "test": (1914, 1941)}


def test_splits_are_contiguous_and_disjoint():
    sp = m5_data.compute_splits(N_DAYS, HORIZON)
    assert sp["train"][1] + 1 == sp["val"][0]
    assert sp["val"][1] + 1 == sp["test"][0]
    assert sp["test"][1] == N_DAYS


def test_subset_selection_ignores_future_sales():
    sales, _, _ = synthetic_m5()
    sp = m5_data.compute_splits(N_DAYS, HORIZON)
    items = m5_data.select_subset(sales, STORES, ["FOODS", "HOBBIES"], 3, sp["train"][1])
    assert "F_4" not in items          # would win if the test window leaked into selection
    assert len(items) == 6
    assert items == m5_data.select_subset(sales, STORES, ["FOODS", "HOBBIES"], 3, sp["train"][1])


def test_explicit_items_override_the_rule():
    sales, _, _ = synthetic_m5()
    items, source = m5_data.resolve_items(sales, STORES, ["FOODS"], 1, 14, ["H_2", "F_3", "H_2"])
    assert items == ["H_2", "F_3"]     # order kept, duplicates removed
    assert "explicit" in source


def test_explicit_items_are_validated():
    sales, _, _ = synthetic_m5()
    with pytest.raises(ValueError, match="NOPE_1"):
        m5_data.resolve_items(sales, STORES, ["FOODS"], 1, 14, ["F_1", "NOPE_1"])
    one_store_only = sales[~((sales["item_id"] == "F_1") & (sales["store_id"] == "S2"))]
    with pytest.raises(ValueError, match="F_1"):
        m5_data.resolve_items(one_store_only, STORES, ["FOODS"], 1, 14, ["F_1"])


def test_to_long_drops_unpriced_rows_and_assigns_splits():
    sales, cal, prices = synthetic_m5()
    sp = m5_data.compute_splits(N_DAYS, HORIZON)
    long = m5_data.assign_split(m5_data.to_long(sales, cal, prices, ["F_1", "H_1"], STORES), sp)
    assert long["sell_price"].notna().all()
    assert long.loc[long["item_id"] == "F_1", "d_num"].min() == 15        # first priced week starts at day 15
    assert long.loc[long["item_id"] == "H_1", "d_num"].min() == 1
    assert set(long["split"]) == {"train", "val", "test"}
    summary = m5_data.summarize(long)
    assert summary["series"] == 4
    assert summary["rows"] == len(long)
