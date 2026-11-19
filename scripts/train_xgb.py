"""Train the shared XGBoost demand model (train split), pick it on validation, compare with the baselines.

Usage (after scripts/prepare_m5_subset.py):
    python scripts/train_xgb.py

Writes:
    models/xgb_demand.json, models/xgb_demand_meta.json   NOT committed (CONTRIBUTING: share via team Drive)
    results/forecast_xgb_val.json                          committed: metrics vs baselines on VALIDATION
The test split is never scored here (CONTRIBUTING rule 2: test exactly once, at the end).
"""
import json
from pathlib import Path

import yaml

from supplyiq.forecasting import m5_data
from supplyiq.forecasting.features import FEATURES
from supplyiq.forecasting.training import run_training

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((ROOT / "configs/forecast.yaml").read_text())
    params = yaml.safe_load((ROOT / "configs/forecast_model.yaml").read_text())["model"]
    ref = json.loads((ROOT / "data/reference/m5_subset.json").read_text())
    splits = {k: tuple(v) for k, v in ref["splits_day_index_inclusive"].items()}
    long = m5_data.load_table(ROOT / cfg["data"]["processed_dir"] / "m5_subset_long")

    model, results = run_training(long, splits, int(ref["n_days"]), params)

    for name in ("xgboost", "naive", "seasonal_naive_7", "moving_average_28"):
        r = results[name]
        print(f"{name:20s} MAE={r['mae']:.4f} RMSE={r['rmse']:.4f} WAPE={r['wape']:.4f} (n={r['n']})")
    print(f"best baseline: {results['best_baseline']}; "
          f"WAPE improvement of XGBoost over it: {results['wape_improvement_vs_best_baseline']:+.4f}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model.save_model(str(models_dir / "xgb_demand.json"))
    meta = {"features": FEATURES, "model_params": params, "splits": ref["splits_day_index_inclusive"],
            "n_days": ref["n_days"], "items": ref["items"], "stores": ref["stores"]}
    (models_dir / "xgb_demand_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    out = ROOT / "results" / "forecast_xgb_val.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"split": "val", "splits": ref["splits_day_index_inclusive"], "results": results},
                              indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} and models/xgb_demand*.json")


if __name__ == "__main__":
    main()
