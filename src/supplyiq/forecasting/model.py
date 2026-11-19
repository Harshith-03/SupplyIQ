"""Thin wrapper around the demand model. XGBoost is imported lazily so importing this module never
needs the XGBoost/OpenMP library (other tests and tools can run without it)."""
from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def make_xgb_regressor(params: dict):
    xgb = importlib.import_module("xgboost")
    return xgb.XGBRegressor(**params)


def train_model(train: pd.DataFrame, val: pd.DataFrame, features: list[str], params: dict,
                model_factory=make_xgb_regressor):
    """Fit on `train`; the validation split is used only for early stopping / model selection.

    `model_factory(params)` must return an object with fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    and predict(X); tests pass a light stand-in, production uses XGBRegressor.
    """
    model = model_factory(params)
    model.fit(
        train[features], train["sales"],
        eval_set=[(val[features], val["sales"])],
        verbose=False,
    )
    return model


def predict(model, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Demand cannot be negative."""
    return np.clip(np.asarray(model.predict(df[features]), dtype=float), 0.0, None)
