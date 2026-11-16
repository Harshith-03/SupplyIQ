"""Build Dataset 5: agent evaluation benchmark.

Ground truth is computed with supplyiq/data/reference_formulas.py (no hand arithmetic).
Task values are hand-chosen and DISJOINT from the NLP-dataset generator (hand-written queries).

Grading types:
  numeric        - closed-form answer; compare agent's final numbers to `expected` within rel_tol
  forecast_dep   - depends on the forecast tool output; grader recomputes `reference_fn(args with $step1.* filled
                   from the agent's actual forecast output)`; also checks forecast quality vs M5 holdout (fill later)
  forecast_acc   - forecast itself; graded against M5 holdout actuals (WAPE) -> filled by eval harness
  rubric         - live web info; graded on criteria only (no fixed numeric truth)
  behavior       - edge case; graded on expected behaviour (ask for info / refuse / report error), never fabricate
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from supplyiq.data import reference_formulas as R

T = []  # tasks


def add(tier, query, intent, tools, calls, grading, criteria, notes=""):
    T.append({"tier": tier, "query": query, "intent": intent, "required_tools": sorted(tools),
              "expected_tool_calls": calls, "grading": grading, "completion_criteria": criteria, "notes": notes})


def calc(op, **p):
    return {"tool": "inventory_calculator", "args": {"operation": op, "params": p}}


def py(task, **p):
    return {"tool": "python_analysis", "args": {"task": task, "params": p}}


def fc(item, store, h):
    return {"tool": "forecast_demand", "args": {"item_id": item, "store_id": store, "horizon_days": h}}


def ws(q):
    return {"tool": "web_search", "args": {"query": q}}


def num(expected, rel_tol=0.01, abs_tol=None):
    g = {"type": "numeric", "expected": expected, "rel_tol": rel_tol}
    if abs_tol is not None:
        g["abs_tol"] = abs_tol
    return g


CALC, PY, FC, WS = "calculator", "python", "forecast", "web_search"
DF, IO, SS, RA, CA, PO = ("demand_forecasting", "inventory_optimization", "supplier_selection",
                          "risk_analysis", "cost_analysis", "procurement_optimization")

# =============== TIER 1: single tool, single call ===============
v = R.reorder_point(120, 9, 150)
add("single_tool", "What is the reorder point for an item with demand of 120 units per day, a 9-day lead time and 150 units of safety stock?",
    IO, [CALC], [calc("reorder_point", avg_daily_demand=120, lead_time_days=9, safety_stock=150)], num({"reorder_point": v}),
    ["Calls calculator with correct reorder_point params", f"Final answer states reorder point ≈ {v:.0f} units"])
v = R.safety_stock(0.95, 18, 16)
add("single_tool", "How much safety stock is needed for a 95% service level if daily demand has a standard deviation of 18 units and lead time is 16 days?",
    IO, [CALC], [calc("safety_stock", service_level=0.95, std_daily_demand=18, lead_time_days=16)], num({"safety_stock": v}),
    ["Uses z for 95% (≈1.645)", f"Final answer ≈ {v:.1f} units"])
v = R.eoq(24000, 75, 3.2)
add("single_tool", "Calculate the economic order quantity for annual demand of 24,000 units, an ordering cost of $75 per order and a holding cost of $3.20 per unit per year.",
    IO, [CALC], [calc("eoq", annual_demand=24000, order_cost=75, holding_cost_per_unit=3.2)], num({"eoq": v}),
    [f"Final answer EOQ ≈ {v:.0f} units"])
v = R.days_of_supply(2400, 85)
add("single_tool", "We have 2,400 units on hand and sell about 85 per day. How many days of supply do we have?",
    IO, [CALC], [calc("days_of_supply", on_hand=2400, avg_daily_demand=85)], num({"days_of_supply": v}),
    [f"Final answer ≈ {v:.1f} days"])
v = R.annual_holding_cost(650, 12.5, 0.22)
add("single_tool", "Average inventory is 650 units, each costing $12.50, and the annual holding rate is 22%. What is the annual holding cost?",
    CA, [CALC], [calc("annual_holding_cost", average_inventory_units=650, unit_cost=12.5, holding_rate=0.22)], num({"annual_holding_cost": v}),
    [f"Final answer = ${v:,.2f}"])
v = R.inventory_turnover(840000, 210000)
add("single_tool", "Cost of goods sold was $840,000 last year and average inventory value was $210,000. What is our inventory turnover?",
    IO, [CALC], [calc("inventory_turnover", cogs=840000, average_inventory_value=210000)], num({"inventory_turnover": v}),
    [f"Final answer = {v:.1f} turns per year"])
v = R.total_annual_cost(12000, 800, 60, 2.4, 7)
add("single_tool", "Annual demand is 12,000 units at $7 each. If we order 800 units at a time, ordering costs $60 per order and holding costs $2.40 per unit per year, what is the total annual cost?",
    CA, [CALC], [calc("total_annual_cost", annual_demand=12000, order_quantity=800, order_cost=60, holding_cost_per_unit=2.4, unit_cost=7)],
    num({"total_annual_cost": v}), [f"Final answer = ${v:,.2f}"])
v = R.order_frequency(18000, 1500)
add("single_tool", "How many orders per year do we place if annual demand is 18,000 units and each order is 1,500 units?",
    PO, [CALC], [calc("order_frequency", annual_demand=18000, order_quantity=1500)], num({"order_frequency": v}),
    [f"Final answer = {v:.0f} orders per year"])
v = R.stockout_cost(340, 4.75)
add("single_tool", "We ran short by 340 units and earn $4.75 margin per unit. What did the stockout cost us?",
    CA, [CALC], [calc("stockout_cost", units_short=340, unit_margin=4.75)], num({"stockout_cost": v}),
    [f"Final answer = ${v:,.2f}"])
v = R.landed_cost(6.4, 2500, 1200)
add("single_tool", "What is the total landed cost of buying 2,500 units at $6.40 each with $1,200 shipping, and what is that per unit?",
    CA, [CALC], [calc("landed_cost", unit_price=6.4, quantity=2500, shipping_cost=1200)], num(v),
    [f"Total = ${v['total_cost']:,.2f}", f"Per unit = ${v['cost_per_unit']:.2f}"])

sd = R.simulate_delay(900, 60, 10, 7)
add("single_tool", "A shipment of an item is delayed by 7 days. We have 900 units, sell 60 per day, and the normal lead time is 10 days. Will we stock out, and how many units short?",
    RA, [PY], [py("simulate_delay", on_hand=900, daily_demand=60, lead_time_days=10, delay_days=7)], num(
        {"stockout_day": sd["stockout_day"], "shortage_units_delayed": sd["shortage_units_delayed"]}),
    ["States stockout occurs around day 15", f"States shortage ≈ {sd['shortage_units_delayed']:.0f} units"])
sups = [{"name": "Supplier A", "unit_price": 4.80, "lead_time_days": 12}, {"name": "Supplier B", "unit_price": 4.35, "lead_time_days": 21},
        {"name": "Supplier C", "unit_price": 4.60, "lead_time_days": 9}]
r = R.compare_suppliers(sups, "lowest_unit_price")
add("single_tool", "Supplier A charges $4.80 per unit with a 12-day lead time, Supplier B $4.35 with 21 days, and Supplier C $4.60 with 9 days. Which supplier has the lowest price?",
    SS, [PY], [py("compare_suppliers", suppliers=sups, criterion="lowest_unit_price")], {"type": "numeric", "expected_text": {"best_supplier": r["best"]}},
    [f"Names {r['best']} as cheapest"])
r = R.compare_suppliers(sups, "weighted", weights={"price": 0.6, "lead_time": 0.4})
add("single_tool", "Using the same three suppliers (A: $4.80/12 days, B: $4.35/21 days, C: $4.60/9 days), which is best if price counts for 60% and lead time for 40%? Score each as weight×(best price/price) plus weight×(best lead time/lead time).",
    SS, [PY], [py("compare_suppliers", suppliers=sups, criterion="weighted", weights={"price": 0.6, "lead_time": 0.4})],
    {"type": "numeric", "expected_text": {"best_supplier": r["best"]}, "expected": {f"score_{k}": x for k, x in r["scores"].items()}, "rel_tol": 0.01},
    [f"Names {r['best']} as best", "Reports scores for all three suppliers"])
bp = R.best_price_break(20000, 50, 0.2, [(1, 5.00), (1000, 4.80), (2500, 4.65)])
add("single_tool", "A supplier charges $5.00 per unit below 1,000 units, $4.80 from 1,000 units and $4.65 from 2,500 units. Annual demand is 20,000, ordering cost is $50 and the holding rate is 20% of price. Which price break minimizes total annual cost?",
    PO, [PY], [py("best_price_break", annual_demand=20000, order_cost=50, holding_rate=0.2, breaks=[[1, 5.0], [1000, 4.8], [2500, 4.65]])],
    {"type": "numeric", "expected": {"best_unit_price": bp["best"]["unit_price"], "best_order_qty": bp["best"]["order_qty"], "best_total_cost": bp["best"]["total_cost"]}, "rel_tol": 0.02},
    [f"Selects the ${bp['best']['unit_price']} price break", f"Order quantity ≈ {bp['best']['order_qty']:.0f}"])
v = R.hhi_concentration([50, 30, 20])
add("single_tool", "Three suppliers provide 50%, 30% and 20% of our volume for an item. What is the HHI concentration index?",
    RA, [PY], [py("hhi_concentration", shares_pct=[50, 30, 20])], num({"hhi": v}),
    [f"Final answer = {v:.0f} (on the 0-10,000 scale)", "Notes this indicates a highly concentrated supply base"])
v = R.stockout_probability_normal(1500, 100, 25, 14)
add("single_tool", "We hold 1,500 units, demand averages 100 per day with a standard deviation of 25, and lead time is 14 days. What is the probability of a stockout during lead time?",
    RA, [PY], [py("monte_carlo_stockout", on_hand=1500, mean_daily_demand=100, std_daily_demand=25, lead_time_days=14)],
    num({"stockout_probability": v}, rel_tol=0.0, abs_tol=0.03),
    [f"Probability ≈ {v:.3f} (±0.03; Monte Carlo noise allowed)"], notes="Analytic normal approximation; MC result must be within abs_tol.")

# forecasts (graded vs M5 holdout actuals)
for item, store, h, q in [
    ("FOODS_3_090", "CA_1", 7, "Predict demand for FOODS_3_090 at store CA_1 over the next 7 days."),
    ("HOBBIES_1_004", None, 28, "How many units of HOBBIES_1_004 will we sell across all stores over the next 28 days?"),
    ("HOUSEHOLD_1_118", "TX_2", 14, "What is the expected daily demand for HOUSEHOLD_1_118 at store TX_2 over the next 14 days?")]:
    add("single_tool", q, DF, [FC], [fc(item, store, h)],
        {"type": "forecast_acc", "metric": "WAPE", "reference": "m5_holdout_actuals", "max_wape": None},
        ["Calls forecast_demand with correct item_id/store_id/horizon_days", "Reports forecast numbers exactly as returned by the tool"],
        notes="max_wape to be set after XGBoost baseline results (e.g. seasonal-naive WAPE). Verify ids exist in the M5 subset.")

# web (rubric)
add("single_tool", "What are current average container freight rates from China to the US West Coast?", CA, [WS], [ws("container freight rates China to US West Coast current")],
    {"type": "rubric"}, ["Calls web_search", "Gives a rate or range with source and date", "Does not state a number absent from search results"])
add("single_tool", "Find reputable suppliers of corrugated shipping boxes in Mexico and their typical lead times.", SS, [WS],
    [ws("corrugated shipping boxes suppliers Mexico lead time")], {"type": "rubric"},
    ["Calls web_search", "Lists at least 2 named suppliers from results", "Lead times only if present in sources, else says unknown"])
add("single_tool", "Are there any port strikes or congestion affecting shipments from Vietnam right now?", RA, [WS],
    [ws("port congestion strike delays shipments from Vietnam current")], {"type": "rubric"},
    ["Calls web_search", "Summarizes current status with dated sources", "States uncertainty if sources conflict or are old"])

# =============== TIER 2: multi-tool (forecast -> calc/python) ===============
ITEM, STORE = "FOODS_3_586", "CA_2"
add("multi_tool", f"Forecast demand for {ITEM} at store {STORE} over the next 14 days and calculate its reorder point with a 6-day lead time and 80 units of safety stock.",
    IO, [FC, CALC], [fc(ITEM, STORE, 14), calc("reorder_point", avg_daily_demand="$step1.mean_daily_demand", lead_time_days=6, safety_stock=80)],
    {"type": "forecast_dep", "reference_fn": "reorder_point", "reference_args": {"avg_daily_demand": "$step1.mean_daily_demand", "lead_time_days": 6, "safety_stock": 80}, "rel_tol": 0.01},
    ["Calls forecast first", "Passes the forecast mean daily demand (not a made-up number) into the calculator", "Final ROP consistent with the forecast output"])
add("multi_tool", f"Forecast next month's demand for {ITEM} across all stores and recommend how much to order, given 1,200 units on hand and 150 units of safety stock.",
    PO, [FC, CALC], [fc(ITEM, None, 28), calc("order_quantity", target_demand="$step1.total_demand", on_hand=1200, safety_stock=150)],
    {"type": "forecast_dep", "reference_fn": "order_quantity", "reference_args": {"target_demand": "$step1.total_demand", "on_hand": 1200, "safety_stock": 150}, "rel_tol": 0.01},
    ["Uses horizon 28 days for 'next month'", "order qty = forecast + safety stock − on hand (floored at 0)", "States the assumption behind 'next month'"])
add("multi_tool", f"Forecast {ITEM} at store {STORE} over the next 7 days, then tell me how many days until I must reorder with 900 units on hand, a 5-day lead time and 60 units of safety stock.",
    PO, [FC, CALC], [fc(ITEM, STORE, 7), calc("days_until_reorder", on_hand=900, avg_daily_demand="$step1.mean_daily_demand", lead_time_days=5, safety_stock=60)],
    {"type": "forecast_dep", "reference_fn": "days_until_reorder", "reference_args": {"on_hand": 900, "avg_daily_demand": "$step1.mean_daily_demand", "lead_time_days": 5, "safety_stock": 60}, "rel_tol": 0.01},
    ["Calls forecast then calculator", "Reports days until reorder (0 if already below ROP) consistent with forecast"])
add("multi_tool", "Forecast demand for HOUSEHOLD_1_334 over the next 28 days and estimate the cost of holding that inventory at $9.50 per unit and a 20% annual rate, assuming it is consumed evenly over the period.",
    CA, [FC, CALC], [fc("HOUSEHOLD_1_334", None, 28), calc("period_holding_cost", average_inventory_units="$step1.total_demand / 2", unit_cost=9.5, holding_rate=0.2, days=28)],
    {"type": "forecast_dep", "reference_fn": "period_holding_cost", "reference_args": {"average_inventory_units": "$step1.total_demand / 2", "unit_cost": 9.5, "holding_rate": 0.2, "days": 28}, "rel_tol": 0.01},
    ["Average inventory = total forecast / 2", "Cost scaled by 28/365"])
add("multi_tool", f"Forecast {ITEM} at store {STORE} over the next 28 days and calculate the EOQ using an ordering cost of $40 and holding cost of $1.80 per unit per year, annualizing the forecast.",
    IO, [FC, CALC], [fc(ITEM, STORE, 28), calc("eoq", annual_demand="$step1.mean_daily_demand * 365", order_cost=40, holding_cost_per_unit=1.8)],
    {"type": "forecast_dep", "reference_fn": "eoq", "reference_args": {"annual_demand": "$step1.mean_daily_demand * 365", "order_cost": 40, "holding_cost_per_unit": 1.8}, "rel_tol": 0.01},
    ["Annualizes via mean daily demand × 365", "EOQ consistent with forecast output"])
add("multi_tool", "Forecast demand for HOBBIES_1_369 over the next 14 days. If a shipment is delayed by 5 days with 700 units on hand and a 7-day lead time, will we stock out?",
    RA, [FC, PY], [fc("HOBBIES_1_369", None, 14), py("simulate_delay", on_hand=700, daily_demand="$step1.mean_daily_demand", lead_time_days=7, delay_days=5)],
    {"type": "forecast_dep", "reference_fn": "simulate_delay", "reference_args": {"on_hand": 700, "daily_demand": "$step1.mean_daily_demand", "lead_time_days": 7, "delay_days": 5}, "rel_tol": 0.01},
    ["Uses forecast mean demand in the delay simulation", "Clear yes/no on stockout with shortage units"])
add("multi_tool", f"Forecast {ITEM} at store {STORE} over the next 14 days, then stress test a 30% demand increase with 500 units on hand and a 6-day lead time.",
    RA, [FC, PY], [fc(ITEM, STORE, 14), py("scenario_demand_shock", daily_demand="$step1.mean_daily_demand", shock_pct=30, on_hand=500, lead_time_days=6)],
    {"type": "forecast_dep", "reference_fn": "scenario_demand_shock", "reference_args": {"daily_demand": "$step1.mean_daily_demand", "shock_pct": 30, "on_hand": 500, "lead_time_days": 6}, "rel_tol": 0.01},
    ["Shocked demand = forecast × 1.3", "Reports days of cover and shortage units"])
add("multi_tool", f"Forecast demand for {ITEM} at store {STORE} over the next 7 days and identify which weekday will peak.",
    DF, [FC, PY], [fc(ITEM, STORE, 7), py("peak_day", data="$step1.daily_forecast")],
    {"type": "forecast_dep", "reference_fn": "peak_day", "reference_args": {"data": "$step1.daily_forecast"}},
    ["Peak weekday matches argmax of the returned daily forecast"])

# =============== TIER 3: multi-step (dependent calls) ===============
e = R.eoq(30000, 90, 4.0)
tc = R.total_annual_cost(30000, e, 90, 4.0, 0)
add("multi_step", "Compute the EOQ for annual demand of 30,000 units, ordering cost $90 and holding cost $4 per unit-year, and then the total annual ordering plus holding cost at that order size.",
    CA, [CALC], [calc("eoq", annual_demand=30000, order_cost=90, holding_cost_per_unit=4.0),
                 calc("total_annual_cost", annual_demand=30000, order_quantity="$step1.eoq", order_cost=90, holding_cost_per_unit=4.0, unit_cost=0)],
    num({"eoq": e, "total_annual_cost": tc}), [f"EOQ ≈ {e:.0f}", f"Total cost ≈ ${tc:,.0f}", "Uses EOQ output as the quantity in step 2"])
ss = R.safety_stock(0.97, 22, 12)
rp = R.reorder_point(140, 12, ss)
add("multi_step", "Find safety stock for a 97% service level with demand standard deviation of 22 per day and a 12-day lead time, then the reorder point at average demand of 140 per day.",
    IO, [CALC], [calc("safety_stock", service_level=0.97, std_daily_demand=22, lead_time_days=12),
                 calc("reorder_point", avg_daily_demand=140, lead_time_days=12, safety_stock="$step1.safety_stock")],
    num({"safety_stock": ss, "reorder_point": rp}), [f"Safety stock ≈ {ss:.1f}", f"Reorder point ≈ {rp:.1f}"])
e = R.eoq(48000, 110, 2.5)
f_ = R.order_frequency(48000, e)
add("multi_step", "How many orders per year does the EOQ imply for annual demand of 48,000, ordering cost $110 and holding cost $2.50 per unit-year?",
    PO, [CALC], [calc("eoq", annual_demand=48000, order_cost=110, holding_cost_per_unit=2.5),
                 calc("order_frequency", annual_demand=48000, order_quantity="$step1.eoq")],
    num({"eoq": e, "order_frequency": f_}), [f"EOQ ≈ {e:.0f}", f"≈ {f_:.1f} orders per year"])
cp = R.compare_order_policies(60000, 150, 1.2, [52, 12])
add("multi_step", "Is it cheaper to order weekly (52 times a year) or monthly (12 times)? Annual demand is 60,000 units, ordering cost $150 and holding cost $1.20 per unit-year.",
    PO, [PY], [py("compare_order_policies", annual_demand=60000, order_cost=150, holding_cost_per_unit=1.2, orders_per_year_options=[52, 12])],
    {"type": "numeric", "expected": {"cost_52": cp["costs"][52], "cost_12": cp["costs"][12]}, "expected_text": {"best_orders_per_year": cp["best_orders_per_year"]}, "rel_tol": 0.01},
    [f"Recommends {cp['best_orders_per_year']} orders per year", "Reports both annual costs"])
ca_ = R.total_annual_cost(36000, 1000, 80, 3.0, 0)
cb_ = R.total_annual_cost(36000, 3000, 80, 3.0, 0)
add("multi_step", "Compare the yearly ordering plus holding cost of ordering 1,000 versus 3,000 units at a time: demand 36,000 per year, ordering cost $80, holding cost $3 per unit-year. Which is cheaper?",
    CA, [CALC], [calc("total_annual_cost", annual_demand=36000, order_quantity=1000, order_cost=80, holding_cost_per_unit=3.0, unit_cost=0),
                 calc("total_annual_cost", annual_demand=36000, order_quantity=3000, order_cost=80, holding_cost_per_unit=3.0, unit_cost=0)],
    num({"cost_q1000": ca_, "cost_q3000": cb_}), [f"Cost at 1,000 = ${ca_:,.0f}", f"Cost at 3,000 = ${cb_:,.0f}", "Names the cheaper option"])
# 3-tool
add("multi_step", f"Forecast demand for {ITEM} at store {STORE} over the next 28 days, calculate the EOQ with ordering cost $55 and holding cost $2/unit-year (annualize the forecast), and simulate stockout risk with 800 units on hand, a 10-day lead time and demand standard deviation of 12.",
    PO, [FC, CALC, PY], [fc(ITEM, STORE, 28), calc("eoq", annual_demand="$step1.mean_daily_demand * 365", order_cost=55, holding_cost_per_unit=2.0),
                         py("monte_carlo_stockout", on_hand=800, mean_daily_demand="$step1.mean_daily_demand", std_daily_demand=12, lead_time_days=10)],
    {"type": "forecast_dep", "reference_fn": ["eoq", "stockout_probability_normal"],
     "reference_args": [{"annual_demand": "$step1.mean_daily_demand * 365", "order_cost": 55, "holding_cost_per_unit": 2.0},
                        {"on_hand": 800, "mean_daily_demand": "$step1.mean_daily_demand", "std_daily_demand": 12, "lead_time_days": 10}], "rel_tol": 0.01, "abs_tol_prob": 0.03},
    ["All three tools used", "Same forecast mean feeds both EOQ and the simulation", "Reports EOQ and stockout probability"])
add("multi_step", "Search for a typical per-unit price of stainless steel water bottles online, then tell me how many I can buy with a $5,000 budget.",
    PO, [WS, CALC], [ws("stainless steel water bottle price per unit wholesale"), calc("budget_coverage", budget=5000, unit_cost="$extract(step1.results, unit_price)")],
    {"type": "rubric", "consistency_check": "units == floor(5000 / price_quoted_by_agent)"},
    ["Price comes from search results with source", "Units = floor(5000 / price) computed with the calculator", "States price assumption clearly"])
add("multi_step", f"Forecast demand for {ITEM} at store {STORE} over the next 14 days, search for packaging suppliers that could deliver within 10 days, and compute the reorder point with a 10-day lead time and 50 units of safety stock.",
    IO, [FC, WS, CALC], [fc(ITEM, STORE, 14), ws("packaging suppliers delivery within 10 days"), calc("reorder_point", avg_daily_demand="$step1.mean_daily_demand", lead_time_days=10, safety_stock=50)],
    {"type": "forecast_dep", "reference_fn": "reorder_point", "reference_args": {"avg_daily_demand": "$step1.mean_daily_demand", "lead_time_days": 10, "safety_stock": 50}, "rel_tol": 0.01, "also_rubric": True},
    ["ROP consistent with forecast", "Supplier info from search results only, with sources"])

# =============== TIER 4: edge cases / error handling ===============
add("edge_case", "Calculate the reorder point for FOODS_3_090 with a 7-day lead time.", IO, [], [],
    {"type": "behavior", "expected_behavior": "ask_clarification_or_state_assumption"},
    ["Notices demand / safety stock are missing", "Either asks for them or runs forecast and states safety stock assumption explicitly", "Does not invent demand numbers"])
add("edge_case", "What is the EOQ for annual demand of 10,000, ordering cost $50 and a holding cost of -$2 per unit per year?", IO, [CALC],
    [calc("eoq", annual_demand=10000, order_cost=50, holding_cost_per_unit=-2)], {"type": "behavior", "expected_behavior": "report_invalid_input"},
    ["Tool returns validation error", "Agent explains negative holding cost is invalid and asks for correction", "No numeric EOQ presented"])
add("edge_case", "Predict demand for item XYZ_9_999 at store CA_1 over the next week.", DF, [FC], [fc("XYZ_9_999", "CA_1", 7)],
    {"type": "behavior", "expected_behavior": "report_item_not_found"}, ["Reports the item is not in the data", "Does not fabricate a forecast"])
add("edge_case", "I need to buy FOODS_3_090 from whoever sells it cheapest - what supplier and price should I go with?", SS, [PY], [py("lookup_supplier_prices", item_id="FOODS_3_090", criterion="lowest_unit_price")],
    {"type": "behavior", "expected_behavior": "no_fabrication"},
    ["Uses supplier catalog data if present, clearly labeled as such", "If no data, says so (or searches the web) instead of inventing prices"],
    notes="Depends on whether team ships a labeled synthetic supplier catalog.")
add("edge_case", "Forecast demand for FOODS_3_090 over the next 2 years.", DF, [FC], [],
    {"type": "behavior", "expected_behavior": "explain_horizon_limit"},
    ["Notes forecast tool supports at most 28 days", "Offers a 28-day forecast or states the limitation; no extrapolated 2-year numbers presented as model output"])
add("edge_case", "How many days of supply do we have if we have 500 units and demand is zero?", IO, [CALC], [calc("days_of_supply", on_hand=500, avg_daily_demand=0)],
    {"type": "behavior", "expected_behavior": "handle_division_by_zero"},
    ["Tool returns a validation error", "Agent explains days of supply is undefined/unbounded when demand is 0"])
add("edge_case", "Write me a poem about warehouses.", IO, [], [], {"type": "behavior", "expected_behavior": "out_of_scope_no_tools"},
    ["Does not call any tool", "Politely declines or briefly redirects to supply-chain scope"])


def main():
    for i, t in enumerate(T):
        t["id"] = f"eval_{i:03d}"
    out = ROOT / "data/eval/agent_benchmark.jsonl"
    with open(out, "w") as f:
        for t in T:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    from collections import Counter
    print(f"wrote {len(T)} tasks ->", out)
    print(dict(Counter(t["tier"] for t in T)))
    print(dict(Counter(t["grading"]["type"] for t in T)))


if __name__ == "__main__":
    main()
