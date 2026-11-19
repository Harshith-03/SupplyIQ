# Design Decision Log

One entry per decision that affects results or fairness. Newest at the bottom.
Format: **ID · date · decision** / why / alternatives considered / owner.

---

**D-001 · 2026-11-15 · Python 3.12 + venv, PyTorch installed separately**
Why: 3.13 still has gaps in ML wheel support. A separate torch install lets GPU and CPU members share one requirements.txt.
Owner: all

**D-002 · 2026-11-15 · Use `sales_train_evaluation.csv` (d_1–d_1941)**
Why: it is the longest history M5 provides. The last 28 days give a natural held-out test horizon that matches "next month."
Owner: all

<!-- Pending team decisions (resolve in Week 1):
  - M5 subset: which stores / departments / how many items?
  - Product naming: how "Product A" in a query maps to an M5 item_id
  - Supplier data: M5 has NONE. Choose web_search-only vs. a small documented assumptions table
-->

## Dataset 4 & 5 design decisions (data/nlp, data/eval)
- **Template-generated bootstrap.** 1,680 queries from 60 scenario families (28 per family) with randomized slot values. Phrasing is regular, so scores on this set will be optimistic; hand-written queries (`source: handwritten`) must be added, mostly to test.
- **Family-level splits.** All paraphrases/instantiations of a family stay in one split (no near-duplicate leakage). Stratified by intent (2 val + 2 test families per intent). Seeded search ensures val/test tool-sets also occur in train.
- **Intent tie-breakers.** Order quantity/date -> procurement; policy parameter (ROP/SS/EOQ) -> inventory; $ figure -> cost; disruption/stockout timing -> risk. "Forecast + reorder point" -> inventory_optimization. Full rules in `docs/annotation_guidelines.md`.
- **Forecast horizon capped at 28 days** (matches M5 evaluation horizon); queries use 1/7/14/28.
- **Supplier numbers in queries are user-supplied hypotheticals** stated in the query text, never tool facts. "Cheapest supplier for item X" queries assume a labelled synthetic supplier catalog or web search (decision pending).
- **Benchmark ground truth** is computed with `supplyiq/data/reference_formulas.py`. Forecast-dependent tasks are graded by recomputing from the agent's own forecast output; forecast quality is graded vs M5 holdout.
- **Item IDs in `data/reference/entity_aliases.json` are placeholders** until verified against the M5 subset (`validate_datasets.py --m5-items`).

## XGBoost forecasting - Step 1 decisions (M5 subset, splits, baselines)
- **Subset:** 3 stores (CA_1, TX_1, WI_1: one per state). Items = EITHER an explicit list in `configs/forecast.yaml` (`subset.items`, used to align with the synthetic inventory dataset) OR, if that list is empty, the top 10 items per category by units sold in those stores during the TRAIN window only (ties by item_id). The chosen ids and how they were chosen are committed in `data/reference/m5_subset.json`; NLP queries and the agent benchmark must use these ids.
- **Chronological splits on `sales_train_evaluation.csv` (1,941 days):** train d_1..d_1885, val d_1886..d_1913, test d_1914..d_1941 (28-day blocks matching the M5 horizon). Test is scored once, at the end (`run_baselines.py` refuses the test split without `--final`).
- **No future leakage:** item selection ignores val/test days; baselines use only days before the split starts (both unit-tested).
- **Unpriced rows dropped:** NaN `sell_price` means the item was not yet on sale in that store/week, not zero demand. Series with no history before a split's cutoff are skipped by the baselines; XGBoost must be scored on exactly the same rows.
- **Shared forecasting metrics:** MAE, RMSE, WAPE live in `src/supplyiq/eval/forecast_metrics.py`.
- **Baselines:** naive (last value), seasonal-naive (7 days), moving average (28 days).
- **PROPOSED, confirm in Step 2:** lag/rolling features use lags >= 28 days so one model can serve any horizon 1..28 without recursive forecasting.
- **PROPOSED, confirm with team:** the forecast tool forecasts from a fixed cutoff (end of val, d_1913) so answers can be verified against M5 actuals in the agent benchmark. "All stores" (store_id null) = sum over the subset stores.

## XGBoost forecasting - Step 2 decisions (features and training)
- **Direct multi-horizon design (confirms the earlier proposal):** every history feature for target day t uses only sales up to day t-28 (lags 28/35/42/49/56/364 and rolling mean/std/zero-share windows of 7/14/28/56 days that END at t-28). One model therefore forecasts any horizon 1..28 days after a cutoff with no recursive prediction. Trade-off: even for short horizons the most recent 27 days are not used; per-horizon models are a possible later improvement.
- **Known-in-advance inputs:** calendar (weekday, month, day of month, week of year, event flag/type, SNAP flag of the store's state) and `sell_price` / `price_ratio_28` of the target day (M5 publishes future prices). Static codes: category, department, store (fixed lookups, so they are identical across splits).
- **Gaps:** history features are computed on a complete day grid per series, so days missing from the table (item not yet on sale) yield NaN instead of silently shifting a lag. NaN is passed to XGBoost, which handles it natively.
- **Objective:** `reg:tweedie` (variance power 1.1) as the starting point for intermittent non-negative counts; hyperparameters live in `configs/forecast_model.yaml` and are tuned on VALIDATION only. Predictions are clipped at 0.
- **Validation only:** `scripts/train_xgb.py` fits on the train split, uses val for early stopping/selection, and compares against the naive baselines on exactly the same rows. It never scores the test split. The trained model goes to `models/` (not committed, share via team Drive); metrics go to `results/forecast_xgb_val.json` (committed).
- **Leakage test:** a unit test tampers with sales after day 40 and checks that no feature for a day before 68 changes.
