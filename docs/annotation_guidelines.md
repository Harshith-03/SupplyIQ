# Annotation Guidelines (Dataset 4 – NLP intent & tool calling)

All members label by these rules so the three NLP models see consistent labels.

## Intent = the user's ultimate goal (exactly one)
| Intent | Use when the goal is… | Typical examples |
|---|---|---|
| demand_forecasting | knowing/analysing demand itself | forecast, seasonality, variability, ABC |
| inventory_optimization | setting inventory *policy parameters* | reorder point, safety stock, EOQ, days of supply, turnover |
| supplier_selection | choosing/vetting a supplier | cheapest, weighted score, reliability, finding suppliers |
| risk_analysis | disruption, uncertainty, stockout timing/probability | delays, Monte Carlo, HHI, stress tests |
| cost_analysis | computing/comparing a **cost** | holding cost, total annual cost, landed cost, freight rates |
| procurement_optimization | deciding **how much / when / from which price tier** to buy | order quantity, next order date, budget, discount breaks, order cadence |

Tie-breakers: (1) If the final deliverable is an order quantity/date → procurement. (2) If it is a policy parameter (ROP, SS, EOQ) → inventory. (3) If it is a $ figure → cost. (4) "Will we run out / what if disrupted" → risk.
"Forecast X and compute reorder point" → **inventory_optimization** (forecast is a means, not the goal).

## Tools = minimal set needed to answer (multi-label)
- `forecast`   – needs future demand for an M5 item (horizon ≤ 28 days).
- `calculator` – closed-form inventory formula with all parameters available.
- `python`     – data analysis on history, simulation, optimisation, comparison across options.
- `web_search` – needs external/live info (prices, suppliers, news, freight rates).
Do NOT add a tool "just in case". Do not tag `forecast` if the user supplies the demand number.

## Tool-call gold format (`tool_calls`)
Ordered list of `{tool, args}`. Values come from the query text; later steps may reference earlier outputs with `"$step1.mean_daily_demand"` (see `configs/label_schema.json`). Only supply args that are stated in the query or derivable from a prior step. Never invent numbers.

## Splits
Split by **family_id** (all paraphrases/instantiations of a scenario stay together). Never re-split randomly by row. Val is for model selection; **test is touched once** for final reporting.

## Adding hand-written queries (strongly recommended)
Write 40–60 queries in your own words (teammates should write for each other), add `"source": "handwritten"`, a unique `family_id` like `hw_017`, and put most of them in **test** (and some in val). This measures generalisation beyond template phrasing.
