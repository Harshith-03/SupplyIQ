"""Score the naive baselines on the validation split (the test split is locked behind --final).

Usage (after scripts/prepare_m5_subset.py):
    python scripts/run_baselines.py                 # validation split
    python scripts/run_baselines.py --split test --final   # ONLY once, at the very end
"""
import argparse
import json
from pathlib import Path

import yaml

from supplyiq.eval import forecast_metrics
from supplyiq.forecasting import baselines, m5_data

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final", action="store_true", help="required to score the test split")
    args = ap.parse_args()
    if args.split == "test" and not args.final:
        raise SystemExit("Refusing to score the test split without --final (CONTRIBUTING rule 2: test once, at the end).")

    cfg = yaml.safe_load((ROOT / "configs/forecast.yaml").read_text())
    ref = json.loads((ROOT / "data/reference/m5_subset.json").read_text())
    splits = {k: tuple(v) for k, v in ref["splits_day_index_inclusive"].items()}
    long = m5_data.load_table(ROOT / cfg["data"]["processed_dir"] / "m5_subset_long")

    results = {}
    for method in baselines.METHODS:
        pred = baselines.baseline_predictions(long, splits, args.split, method)
        overall = forecast_metrics.evaluate(pred["y_true"], pred["y_pred"])
        by_cat = {
            cat: forecast_metrics.wape(g["y_true"], g["y_pred"]) for cat, g in pred.groupby("cat_id")
        }
        results[method] = {**overall, "wape_by_category": by_cat}
        print(f"{method:20s} MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} WAPE={overall['wape']:.4f} (n={overall['n']})")

    out = ROOT / "results" / f"forecast_baselines_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "splits": ref["splits_day_index_inclusive"], "results": results}, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
