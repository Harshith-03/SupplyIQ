"""Train on the train split, select on validation, compare against the naive baselines on the SAME rows."""
from __future__ import annotations

import pandas as pd

from supplyiq.eval import forecast_metrics
from supplyiq.forecasting import baselines
from supplyiq.forecasting.features import FEATURES, build_features
from supplyiq.forecasting.model import make_xgb_regressor, predict, train_model

KEYS = ["item_id", "store_id", "d_num"]


def run_training(
    long: pd.DataFrame,
    splits: dict[str, tuple[int, int]],
    n_days: int,
    model_params: dict,
    model_factory=make_xgb_regressor,
):
    """Returns (model, results). Only train and val are touched; the test split is never scored here."""
    df = build_features(long, n_days)
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"].copy()
    model = train_model(train, val, FEATURES, model_params, model_factory)
    val["y_pred"] = predict(model, val, FEATURES)

    base_preds = {m: baselines.baseline_predictions(long, splits, "val", m) for m in baselines.METHODS}
    keys = base_preds["naive"][KEYS]                      # same row set for every method
    scored = val.merge(keys, on=KEYS, how="inner")

    def block(y_true, y_pred, cats):
        out = forecast_metrics.evaluate(y_true, y_pred)
        out["wape_by_category"] = {
            c: forecast_metrics.wape(g["y_true"], g["y_pred"])
            for c, g in pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "cat": cats}).groupby("cat")
        }
        return out

    results = {"xgboost": block(scored["sales"].to_numpy(), scored["y_pred"].to_numpy(), scored["cat_id"].to_numpy())}
    for name, pred in base_preds.items():
        results[name] = block(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy(), pred["cat_id"].to_numpy())

    best_name = min(baselines.METHODS, key=lambda n: results[n]["wape"])
    results["best_baseline"] = best_name
    results["wape_improvement_vs_best_baseline"] = results[best_name]["wape"] - results["xgboost"]["wape"]
    results["n_features"] = len(FEATURES)
    results["best_iteration"] = getattr(model, "best_iteration", None)
    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        ranked = sorted(zip(FEATURES, importances, strict=True), key=lambda kv: -kv[1])[:15]
        results["top_feature_importances"] = {k: float(v) for k, v in ranked}
    return model, results
