import numpy as np
import pytest

from supplyiq.forecasting import baselines
from supplyiq.forecasting.features import FEATURES, build_features
from supplyiq.forecasting.model import predict, train_model
from supplyiq.forecasting.training import run_training
from tests.forecast_feature_fixtures import synthetic_long_with_calendar
from tests.forecast_fixtures import N_DAYS


class MeanModel:
    """Light stand-in with the same fit/predict contract as XGBRegressor."""

    def __init__(self, params, offset=0.0):
        self.params = params
        self.offset = offset
        self.train_n = None
        self.eval_n = None

    def fit(self, x, y, eval_set=None, verbose=False):
        self.mean_ = float(np.mean(y))
        self.train_n = len(x)
        self.eval_n = len(eval_set[0][0]) if eval_set else None
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_ + self.offset)


def test_validation_is_used_only_as_eval_set():
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    model = train_model(train, val, FEATURES, {}, model_factory=MeanModel)
    assert model.train_n == len(train)
    assert model.eval_n == len(val)


def test_predictions_are_never_negative():
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    model = MeanModel({}, offset=-1000.0).fit(df[FEATURES], df["sales"])
    assert (predict(model, df, FEATURES) == 0).all()


def test_run_training_scores_xgboost_and_baselines_on_the_same_rows():
    long, splits = synthetic_long_with_calendar()
    model, results = run_training(long, splits, N_DAYS, {}, model_factory=MeanModel)
    n = results["xgboost"]["n"]
    assert all(results[m]["n"] == n for m in baselines.METHODS)
    assert results["best_baseline"] in baselines.METHODS
    assert set(results["xgboost"]["wape_by_category"]) == {"FOODS", "HOBBIES"}
    assert results["n_features"] == len(FEATURES)
    # the stand-in saw only train rows for fitting and only val rows for evaluation
    n_train = int((build_features(long, N_DAYS)["split"] == "train").sum())
    assert model.train_n == n_train


def test_real_xgboost_fits_and_predicts_non_negative():
    pytest.importorskip("xgboost")
    long, _ = synthetic_long_with_calendar()
    df = build_features(long, N_DAYS)
    train, val = df[df["split"] == "train"], df[df["split"] == "val"]
    params = {
        "objective": "reg:tweedie", "tweedie_variance_power": 1.1, "n_estimators": 20, "learning_rate": 0.1,
        "max_depth": 3, "tree_method": "hist", "eval_metric": "rmse", "early_stopping_rounds": 5,
        "random_state": 0, "n_jobs": 1,
    }
    model = train_model(train, val, FEATURES, params)
    pred = predict(model, val, FEATURES)
    assert len(pred) == len(val)
    assert np.isfinite(pred).all()
    assert (pred >= 0).all()
