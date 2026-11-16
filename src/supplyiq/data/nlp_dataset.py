"""Generate Dataset 4: supply-chain queries with intent, required tools and expected tool calls.

Usage:  python scripts/build_nlp_dataset.py   (settings in configs/nlp_dataset.yaml)

Design:
  * ~60 scenario *families* (10 per intent). Each family = 2-3 phrasings + a function that
    builds the gold tool calls from the SAME slot values used in the text (text/args can't disagree).
  * Splits are made at FAMILY level (stratified by tool-set) -> no paraphrase/near-duplicate leakage.
  * Template data is a bootstrap. Add hand-written queries (data/nlp/handwritten.jsonl) to the test split.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = json.loads((ROOT / "data/reference/entity_aliases.json").read_text())
SCHEMA = json.loads((ROOT / "data/nlp/label_schema.json").read_text())
L2F = SCHEMA["tool_label_to_function"]
F2L = {v: k for k, v in L2F.items()}

HORIZONS = [("tomorrow", 1), ("next week", 7), ("over the next 14 days", 14), ("over the next two weeks", 14),
            ("next month", 28), ("over the next 4 weeks", 28)]
REGIONS = ["China", "Vietnam", "Mexico", "India", "Turkey", "Germany", "Brazil", "Bangladesh"]
PRODUCTS = ["corrugated shipping boxes", "pallet stretch wrap", "lithium-ion batteries", "cotton t-shirts",
            "plastic storage bins", "LED light bulbs", "glass jars", "packing tape", "organic cotton towels",
            "stainless steel water bottles"]
CATEGORIES = ["FOODS", "HOBBIES", "HOUSEHOLD"]


def m(x):  # money text
    return f"{x:.2f}"


# ------------------------------------------------------------------ slots
def sample_slots(rng):
    s = {}
    if rng.random() < 0.3:
        s["item"] = rng.choice(list(CFG["aliases"]))
        s["item_id"] = CFG["aliases"][s["item"]]
    else:
        s["item"] = s["item_id"] = rng.choice(CFG["items"])
    if rng.random() < 0.5:
        s["store_id"] = rng.choice(CFG["stores"])
        s["at_store"] = f" at store {s['store_id']}"
    else:
        s["store_id"], s["at_store"] = None, ""
    s["h"], s["h_days"] = rng.choice(HORIZONS)
    s["d"] = rng.choice(range(10, 400, 5))
    s["sd"] = max(2, round(s["d"] * rng.uniform(0.1, 0.4)))
    s["L"] = rng.randint(2, 30)
    s["L2"] = s["L"] + rng.randint(3, 15)
    s["sl_pct"] = rng.choice([90, 95, 97, 99])
    s["sl"] = s["sl_pct"] / 100
    s["ss"] = rng.choice(range(20, 500, 10))
    s["on_hand"] = rng.choice(range(100, 4000, 50))
    s["D"] = rng.choice(range(2000, 120001, 500))
    s["S"] = rng.choice(range(20, 260, 5))
    s["H"] = rng.choice([0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12])
    s["H2"] = s["H"] + rng.choice([1, 2, 3])
    s["C"] = round(rng.uniform(2, 80), 2)
    s["hr_pct"] = rng.choice([15, 18, 20, 22, 25, 30])
    s["hr"] = s["hr_pct"] / 100
    s["Q"] = rng.choice(range(100, 5001, 50))
    s["Q2"] = s["Q"] + rng.choice(range(100, 1501, 50))
    s["delay"] = rng.randint(2, 21)
    s["budget"] = rng.choice(range(1000, 50001, 500))
    s["cogs"] = rng.choice(range(100000, 5000001, 10000))
    s["avg_inv"] = max(1000, round(s["cogs"] / rng.uniform(2, 12), -3))
    s["short"] = rng.choice(range(10, 2000, 10))
    s["margin"] = round(rng.uniform(0.5, 20), 2)
    s["spend"] = rng.choice(range(10000, 2000001, 5000))
    s["pct"] = rng.choice([3, 5, 7, 8, 10, 12, 15, 20, 25])
    s["up"] = rng.choice([10, 15, 20, 30, 50])
    s["region"] = rng.choice(REGIONS)
    s["prod"] = rng.choice(PRODUCTS)
    s["cat"] = rng.choice(CATEGORIES)
    # three suppliers (clearly hypothetical numbers that appear in the query text)
    base = rng.uniform(2, 60)
    sup = []
    for nm in ["A", "B", "C"]:
        sup.append({"name": f"Supplier {nm}", "unit_price": round(base * rng.uniform(0.85, 1.2), 2),
                    "lead_time_days": rng.randint(4, 40), "shipping_cost": rng.choice(range(100, 3001, 50)),
                    "on_time_rate": rng.randint(78, 99), "moq": rng.choice([50, 100, 250, 500, 1000])})
    s["sup"] = sup
    s["wp"], s["wl"] = rng.choice([(70, 30), (60, 40), (50, 50), (80, 20)])
    a = rng.randint(30, 70)
    b = rng.randint(10, 100 - a - 5)
    s["shares"] = [a, b, 100 - a - b]
    p0 = round(rng.uniform(3, 40), 2)
    t1 = rng.choice([500, 1000, 2000])
    s["p0"], s["p1"], s["p2"] = p0, round(p0 * 0.96, 2), round(p0 * 0.92, 2)
    s["t1"], s["t2"] = t1, int(t1 * 2.5)
    # preformatted supplier phrases
    if rng.random() < 0.5:
        first = ", ".join(f"{x['name']} charges ${m(x['unit_price'])} per unit with a {x['lead_time_days']}-day lead time" for x in sup[:2])
        s["sup_price_lead"] = first + f" and {sup[2]['name']} charges ${m(sup[2]['unit_price'])} with {sup[2]['lead_time_days']} days"
    else:
        s["sup_price_lead"] = "; ".join(f"{x['name']}: ${m(x['unit_price'])}/unit, {x['lead_time_days']} days lead time" for x in sup)
    s["sup_ship"] = "; ".join(f"{x['name']}: ${m(x['unit_price'])}/unit plus ${x['shipping_cost']} shipping" for x in sup[:2])
    s["sup_otd"] = ", ".join(f"{x['name']} {x['on_time_rate']}%" for x in sup)
    s["sup_moq"] = "; ".join(f"{x['name']} (MOQ {x['moq']}, ${m(x['unit_price'])}/unit)" for x in sup[:2])
    s["shares_txt"] = ", ".join(f"{x}%" for x in s["shares"])
    return s


# ------------------------------------------------------------------ call builders
def fc(s):
    return {"tool": "forecast_demand", "args": {"item_id": s["item_id"], "store_id": s["store_id"], "horizon_days": s["h_days"]}}


def calc(op, **params):
    return {"tool": "inventory_calculator", "args": {"operation": op, "params": params}}


def py(task, **params):
    return {"tool": "python_analysis", "args": {"task": task, "params": params}}


def ws(q):
    return {"tool": "web_search", "args": {"query": q}}


def sups(s, keys, n=3):
    return [{"name": x["name"], **{k: x[k] for k in keys}} for x in s["sup"][:n]]


MD = "$step1.mean_daily_demand"
TD = "$step1.total_demand"

# ------------------------------------------------------------------ families
# (family_id, intent, [phrasings], calls_fn)
F = []


def fam(fid, intent, variants, fn):
    F.append((fid, intent, variants, fn))


DF, IO, SS, RA, CA, PO = ("demand_forecasting", "inventory_optimization", "supplier_selection",
                          "risk_analysis", "cost_analysis", "procurement_optimization")

# ---- demand forecasting
fam("df01", DF, ["Predict demand for {item}{at_store} {h}.", "How many units of {item} will we sell {h}{at_store}?",
                 "Forecast sales for {item}{at_store} {h}."], lambda s: [fc(s)])
fam("df02", DF, ["What is the expected daily demand for {item}{at_store} {h}?",
                 "Give me the daily demand forecast for {item}{at_store} {h}.",
                 "I need the day-by-day sales outlook for {item}{at_store} {h}."], lambda s: [fc(s)])
fam("df03", DF, ["Forecast demand for {item}{at_store} {h} and tell me which weekday will peak.",
                 "{item}{at_store}: predict sales {h} and identify the busiest day of the week."],
    lambda s: [fc(s), py("peak_day", data="$step1.daily_forecast")])
fam("df04", DF, ["Forecast {item}{at_store} {h} and compare it with the same period last year.",
                 "How does predicted demand for {item}{at_store} {h} compare to the same period last year?"],
    lambda s: [fc(s), py("compare_with_history", item_id=s["item_id"], store_id=s["store_id"],
                         data="$step1.daily_forecast", baseline="same_period_last_year")])
fam("df05", DF, ["How volatile is historical demand for {item}{at_store}? Compute the coefficient of variation.",
                 "Calculate the demand variability (CV) of {item}{at_store} from past sales.",
                 "What is the coefficient of variation of the sales history for {item}{at_store}?"],
    lambda s: [py("demand_variability", item_id=s["item_id"], store_id=s["store_id"])])
fam("df06", DF, ["Run an ABC analysis on the {cat} category based on historical sales{at_store}.",
                 "Classify {cat} items into A, B and C groups by sales volume{at_store}."],
    lambda s: [py("abc_analysis", category=s["cat"], store_id=s["store_id"])])
fam("df07", DF, ["Show the weekly seasonality pattern in historical sales of {item}{at_store}.",
                 "Which days of the week does {item}{at_store} usually sell best, based on past data?"],
    lambda s: [py("seasonality_profile", item_id=s["item_id"], store_id=s["store_id"])])
fam("df08", DF, ["Forecast {item}{at_store} {h} and check online for upcoming holidays or events that might shift demand.",
                 "Predict {item} demand{at_store} {h}, and search the web for retail events that could affect it."],
    lambda s: [fc(s), ws("upcoming US holidays and retail events affecting consumer demand")])
fam("df09", DF, ["Estimate total units of {item}{at_store} we'll sell {h}.",
                 "What total demand should we expect for {item}{at_store} {h}?"], lambda s: [fc(s)])
fam("df10", DF, ["Forecast {item}{at_store} {h} and run a what-if where demand ends up {up}% higher.",
                 "Predict demand for {item}{at_store} {h}, then show what a {up}% demand spike would look like."],
    lambda s: [fc(s), py("scenario_demand_shock", daily_demand=MD, shock_pct=s["up"], on_hand=None, lead_time_days=None)])

# ---- inventory optimization
fam("io01", IO, ["Calculate the reorder point for {item}: average demand {d} units/day, lead time {L} days, safety stock {ss} units.",
                 "What reorder point should I use for {item} with {d} units/day demand, {L} days lead time and {ss} units of safety stock?",
                 "Reorder point for {item}? Demand is {d} per day, lead time {L} days, safety stock {ss}."],
    lambda s: [calc("reorder_point", avg_daily_demand=s["d"], lead_time_days=s["L"], safety_stock=s["ss"])])
fam("io02", IO, ["What safety stock do I need for a {sl_pct}% service level with daily demand standard deviation {sd} and a lead time of {L} days?",
                 "Compute safety stock: service level {sl_pct}%, demand std dev {sd}/day, lead time {L} days."],
    lambda s: [calc("safety_stock", service_level=s["sl"], std_daily_demand=s["sd"], lead_time_days=s["L"])])
fam("io03", IO, ["Calculate the economic order quantity: annual demand {D} units, ordering cost ${S} per order, holding cost ${H} per unit per year.",
                 "What is the EOQ if we sell {D} units a year, each order costs ${S} to place and holding costs ${H} per unit annually?",
                 "Compute EOQ for D={D}, S=${S}, H=${H}."],
    lambda s: [calc("eoq", annual_demand=s["D"], order_cost=s["S"], holding_cost_per_unit=s["H"])])
fam("io04", IO, ["Forecast demand for {item}{at_store} {h} and calculate its reorder point with a {L}-day lead time and {ss} units of safety stock.",
                 "Predict {item} demand{at_store} {h}, then work out the reorder point given {L} days lead time and safety stock of {ss}."],
    lambda s: [fc(s), calc("reorder_point", avg_daily_demand=MD, lead_time_days=s["L"], safety_stock=s["ss"])])
fam("io05", IO, ["Forecast demand for {item}{at_store} {h} and calculate the EOQ, using ordering cost ${S} and holding cost ${H} per unit per year (annualize the forecast).",
                 "Use the demand forecast for {item}{at_store} {h}, annualized, to compute the economic order quantity with ${S} per order and ${H} holding cost per unit-year."],
    lambda s: [fc(s), calc("eoq", annual_demand=MD + " * 365", order_cost=s["S"], holding_cost_per_unit=s["H"])])
fam("io06", IO, ["I have {on_hand} units of {item} and sell about {d} a day. How many days of supply is that?",
                 "Days of supply for {item}: {on_hand} on hand, {d} units/day demand."],
    lambda s: [calc("days_of_supply", on_hand=s["on_hand"], avg_daily_demand=s["d"])])
fam("io07", IO, ["Simulate the stockout probability for {item} over a {L}-day lead time with {on_hand} units on hand, mean demand {d}/day and standard deviation {sd}.",
                 "Run a Monte Carlo simulation: {on_hand} units in stock, demand {d}±{sd} per day, {L}-day lead time. How likely is a stockout?"],
    lambda s: [py("monte_carlo_stockout", on_hand=s["on_hand"], mean_daily_demand=s["d"], std_daily_demand=s["sd"], lead_time_days=s["L"])])
fam("io08", IO, ["Annual COGS is ${cogs} and average inventory value is ${avg_inv}. What is the inventory turnover?",
                 "Calculate inventory turnover with cost of goods sold ${cogs} and average inventory ${avg_inv}."],
    lambda s: [calc("inventory_turnover", cogs=s["cogs"], average_inventory_value=s["avg_inv"])])
fam("io09", IO, ["How does the reorder point change if lead time varies from {L} to {L2} days? Demand is {d}/day and safety stock {ss}.",
                 "Run a sensitivity analysis of reorder point for lead times between {L} and {L2} days (demand {d}/day, safety stock {ss})."],
    lambda s: [py("sensitivity_analysis", metric="reorder_point", vary="lead_time_days", values=[s["L"], s["L2"]],
                  base={"avg_daily_demand": s["d"], "safety_stock": s["ss"]})])
fam("io10", IO, ["How many orders per year will I place if annual demand is {D} units and I order {Q} units at a time?",
                 "Order frequency: {D} units per year, order size {Q}."],
    lambda s: [calc("order_frequency", annual_demand=s["D"], order_quantity=s["Q"])])

# ---- supplier selection
fam("ss01", SS, ["{sup_price_lead}. Which supplier is cheapest?", "Compare suppliers by unit price - {sup_price_lead}."],
    lambda s: [py("compare_suppliers", suppliers=sups(s, ["unit_price", "lead_time_days"]), criterion="lowest_unit_price")])
fam("ss02", SS, ["{sup_price_lead}. Which is best if price is weighted {wp}% and lead time {wl}%?",
                 "Pick a supplier with a {wp}/{wl} price-to-lead-time weighting. {sup_price_lead}."],
    lambda s: [py("compare_suppliers", suppliers=sups(s, ["unit_price", "lead_time_days"]), criterion="weighted",
                  weights={"price": s["wp"] / 100, "lead_time": s["wl"] / 100})])
fam("ss03", SS, ["Which supplier offers the lowest price for {item}?", "Who is the cheapest supplier for {item}?",
                 "Find the lowest supplier price for {item} in our supplier catalog."],
    lambda s: [py("lookup_supplier_prices", item_id=s["item_id"], criterion="lowest_unit_price")])
fam("ss04", SS, ["Find reputable suppliers of {prod} in {region} and their typical lead times.",
                 "Search for {prod} manufacturers in {region} and how long they take to deliver."],
    lambda s: [ws(f"{s['prod']} suppliers manufacturers {s['region']} lead time")])
fam("ss05", SS, ["For an order of {Q} units, which is cheaper overall - {sup_ship}?",
                 "Total landed cost comparison for {Q} units: {sup_ship}. Which supplier wins?"],
    lambda s: [py("compare_suppliers", suppliers=sups(s, ["unit_price", "shipping_cost"], 2), criterion="lowest_total_cost", quantity=s["Q"])])
fam("ss06", SS, ["Rank suppliers by reliability. On-time delivery: {sup_otd}.", "Which supplier is the most reliable? {sup_otd} on-time delivery."],
    lambda s: [py("compare_suppliers", suppliers=sups(s, ["on_time_rate"]), criterion="highest_on_time_rate")])
fam("ss07", SS, ["Look up current online prices for {prod} and pick the cheapest supplier.",
                 "Search the web for {prod} prices from different suppliers and choose the lowest."],
    lambda s: [ws(f"{s['prod']} wholesale price per unit suppliers"),
               py("compare_suppliers", suppliers="$extract(step1.results, suppliers)", criterion="lowest_unit_price")])
fam("ss08", SS, ["Based on the forecast for {item}{at_store} {h}, which is more economical: {sup_moq}?",
                 "Forecast {item}{at_store} {h} and decide between {sup_moq} using the forecast quantity."],
    lambda s: [fc(s), py("compare_suppliers", suppliers=sups(s, ["unit_price"], 2), criterion="lowest_total_cost", quantity=TD)])
fam("ss09", SS, ["Are there recent news about quality or compliance problems with {prod} suppliers in {region}?",
                 "Check the web for supplier compliance or quality issues affecting {prod} from {region}."],
    lambda s: [ws(f"{s['prod']} supplier quality compliance issues {s['region']} news")])
fam("ss10", SS, ["Forecast demand for {item}{at_store} {h} and find suppliers who could deliver that volume of {prod} within {L} days.",
                 "Predict {item} demand{at_store} {h}, then search for {prod} suppliers able to ship within {L} days."],
    lambda s: [fc(s), ws(f"{s['prod']} suppliers delivery within {s['L']} days bulk orders")])

# ---- risk analysis
fam("ra01", RA, ["What happens to inventory of {item} if a shipment is delayed by {delay} days? We have {on_hand} units, sell {d}/day, normal lead time {L} days.",
                 "Simulate a {delay}-day shipment delay for {item}: {on_hand} units on hand, demand {d} per day, lead time {L} days."],
    lambda s: [py("simulate_delay", on_hand=s["on_hand"], daily_demand=s["d"], lead_time_days=s["L"], delay_days=s["delay"])])
fam("ra02", RA, ["Run a Monte Carlo simulation of stockout risk with demand {d}±{sd} per day and lead time between {L} and {L2} days, starting from {on_hand} units.",
                 "What's the stockout probability if lead time is uncertain ({L}-{L2} days), demand is {d}±{sd}/day and I hold {on_hand} units?"],
    lambda s: [py("monte_carlo_stockout", on_hand=s["on_hand"], mean_daily_demand=s["d"], std_daily_demand=s["sd"],
                  lead_time_min=s["L"], lead_time_max=s["L2"])])
fam("ra03", RA, ["Forecast demand for {item}{at_store} {h}; if a shipment is delayed {delay} days (lead time {L} days, {on_hand} units on hand), will we stock out?",
                 "Predict {item} demand{at_store} {h} and check the stockout risk of a {delay}-day delay, with {on_hand} on hand and a {L}-day lead time."],
    lambda s: [fc(s), py("simulate_delay", on_hand=s["on_hand"], daily_demand=MD, lead_time_days=s["L"], delay_days=s["delay"])])
fam("ra04", RA, ["Are there any port strikes or congestion affecting shipments from {region} right now?",
                 "Check for current port delays on cargo coming from {region}."],
    lambda s: [ws(f"port congestion strike delays shipments from {s['region']} current")])
fam("ra05", RA, ["What is the current risk of supply disruption in {region} from weather or geopolitical events?",
                 "Search for supply chain disruption risks in {region} at the moment."],
    lambda s: [ws(f"supply chain disruption risk {s['region']} weather geopolitical")])
fam("ra06", RA, ["Our supply of {item} comes from three suppliers with shares {shares_txt}. Compute the concentration risk (HHI).",
                 "Calculate the HHI supplier concentration for {item} with volume shares of {shares_txt}."],
    lambda s: [py("hhi_concentration", shares_pct=s["shares"])])
fam("ra07", RA, ["How much safety stock do we need if lead time rises from {L} to {L2} days? Demand std dev {sd}/day, {sl_pct}% service level.",
                 "Recompute safety stock for a longer lead time of {L2} days instead of {L} (std dev {sd}, service level {sl_pct}%)."],
    lambda s: [calc("safety_stock", service_level=s["sl"], std_daily_demand=s["sd"], lead_time_days=s["L2"])])
fam("ra08", RA, ["Forecast {item}{at_store} {h} and test what happens to stockout risk if demand is {up}% above forecast, with {on_hand} units on hand and a {L}-day lead time.",
                 "Stress test: {item} demand{at_store} {h} comes in {up}% higher than predicted, {on_hand} units in stock, {L} days lead time."],
    lambda s: [fc(s), py("scenario_demand_shock", daily_demand=MD, shock_pct=s["up"], on_hand=s["on_hand"], lead_time_days=s["L"])])
fam("ra09", RA, ["How many days until we stock out of {item} with {on_hand} units and demand of {d} per day?",
                 "When will {item} run out? {on_hand} units on hand, {d} sold daily."],
    lambda s: [calc("days_of_supply", on_hand=s["on_hand"], avg_daily_demand=s["d"])])
fam("ra10", RA, ["What is the probability of a stockout for {item}: {on_hand} units on hand, mean demand {d}, std {sd}, lead time {L} days?",
                 "Estimate stockout risk over the {L}-day lead time of {item} given {on_hand} units and demand {d}±{sd} per day."],
    lambda s: [py("monte_carlo_stockout", on_hand=s["on_hand"], mean_daily_demand=s["d"], std_daily_demand=s["sd"], lead_time_days=s["L"])])

# ---- cost analysis
fam("ca01", CA, ["Average inventory is {Q} units at ${C} each with a {hr_pct}% annual holding rate. What is the annual holding cost?",
                 "Compute annual holding cost: {Q} units average inventory, unit cost ${C}, holding rate {hr_pct}%."],
    lambda s: [calc("annual_holding_cost", average_inventory_units=s["Q"], unit_cost=s["C"], holding_rate=s["hr"])])
fam("ca02", CA, ["Total annual inventory cost for demand {D}, order quantity {Q}, ordering cost ${S}, holding cost ${H}/unit/year and unit cost ${C}?",
                 "What does it cost per year to order {Q} units at a time? D={D}, S=${S}, H=${H}, unit cost ${C}."],
    lambda s: [calc("total_annual_cost", annual_demand=s["D"], order_quantity=s["Q"], order_cost=s["S"],
                    holding_cost_per_unit=s["H"], unit_cost=s["C"])])
fam("ca03", CA, ["Compute the EOQ and the total annual cost at that order size: demand {D}/year, ordering cost ${S}, holding cost ${H} per unit-year.",
                 "Find the economic order quantity and what the yearly ordering plus holding cost would be (D={D}, S=${S}, H=${H})."],
    lambda s: [calc("eoq", annual_demand=s["D"], order_cost=s["S"], holding_cost_per_unit=s["H"]),
               calc("total_annual_cost", annual_demand=s["D"], order_quantity="$step1.eoq", order_cost=s["S"],
                    holding_cost_per_unit=s["H"], unit_cost=0)])
fam("ca04", CA, ["Compare yearly ordering plus holding cost if we order {Q} versus {Q2} units at a time (demand {D}, ordering cost ${S}, holding ${H}/unit/year).",
                 "Is ordering {Q} or {Q2} units cheaper per year? D={D}, S=${S}, H=${H}."],
    lambda s: [calc("total_annual_cost", annual_demand=s["D"], order_quantity=s["Q"], order_cost=s["S"], holding_cost_per_unit=s["H"], unit_cost=0),
               calc("total_annual_cost", annual_demand=s["D"], order_quantity=s["Q2"], order_cost=s["S"], holding_cost_per_unit=s["H"], unit_cost=0)])
fam("ca05", CA, ["Forecast demand for {item}{at_store} {h} and estimate the cost of holding that inventory at ${C} per unit and a {hr_pct}% annual rate, assuming it is consumed evenly over the period.",
                 "Predict {item} demand{at_store} {h}, then compute the holding cost for the period (unit cost ${C}, {hr_pct}% annual holding rate, even consumption)."],
    lambda s: [fc(s), calc("period_holding_cost", average_inventory_units=TD + " / 2", unit_cost=s["C"], holding_rate=s["hr"], days=s["h_days"])])
fam("ca06", CA, ["What is the total cost of buying {Q} units at ${C} each with ${S} shipping?",
                 "Landed cost for {Q} units at ${C} per unit plus ${S} freight?"],
    lambda s: [calc("landed_cost", unit_price=s["C"], quantity=s["Q"], shipping_cost=s["S"])])
fam("ca07", CA, ["How sensitive is total annual inventory cost to holding cost between ${H} and ${H2} per unit-year? Demand {D}, order quantity {Q}, ordering cost ${S}.",
                 "Sensitivity analysis: vary holding cost from ${H} to ${H2}/unit/year (D={D}, Q={Q}, S=${S}) and show annual cost."],
    lambda s: [py("sensitivity_analysis", metric="total_annual_cost", vary="holding_cost_per_unit", values=[s["H"], s["H2"]],
                  base={"annual_demand": s["D"], "order_quantity": s["Q"], "order_cost": s["S"], "unit_cost": 0})])
fam("ca08", CA, ["What are current average container freight rates from {region} to the US West Coast?",
                 "Search for the latest sea freight cost per container from {region} to the US."],
    lambda s: [ws(f"container freight rates {s['region']} to US West Coast current")])
fam("ca09", CA, ["We were short {short} units and lose ${margin} margin per unit. What is the stockout cost?",
                 "Stockout cost: {short} units unavailable, margin ${margin} each."],
    lambda s: [calc("stockout_cost", units_short=s["short"], unit_margin=s["margin"])])
fam("ca10", CA, ["Annual spend is ${spend}. What is the cost impact of a {pct}% price increase?",
                 "How much more will we pay per year if prices rise {pct}% on ${spend} of spend?"],
    lambda s: [calc("price_change_impact", annual_spend=s["spend"], pct_change=s["pct"])])

# ---- procurement optimization
fam("po01", PO, ["Forecast next month's demand for {item}{at_store} and recommend how much to order, given {on_hand} units on hand and {ss} safety stock.",
                 "How much {item}{at_store} should I order to cover demand {h}? I hold {on_hand} units and want {ss} as safety stock."],
    lambda s: [fc(s), calc("order_quantity", target_demand=TD, on_hand=s["on_hand"], safety_stock=s["ss"])])
fam("po02", PO, ["With a budget of ${budget} and unit cost of ${C}, how many units can I buy and how many days of demand ({d}/day) does that cover?",
                 "Budget ${budget}, unit cost ${C}, demand {d} per day: how many units and how many days covered?"],
    lambda s: [calc("budget_coverage", budget=s["budget"], unit_cost=s["C"], avg_daily_demand=s["d"])])
fam("po03", PO, ["Supplier offers ${p0} per unit under {t1} units, ${p1} from {t1} units and ${p2} from {t2} units. Annual demand {D}, order cost ${S}, holding rate {hr_pct}%. Which price break minimizes total cost?",
                 "Quantity discount decision: ${p0} (<{t1}), ${p1} ({t1}+), ${p2} ({t2}+). D={D}/year, S=${S}, holding {hr_pct}% of price. Best order quantity?"],
    lambda s: [py("best_price_break", annual_demand=s["D"], order_cost=s["S"], holding_rate=s["hr"],
                  breaks=[[1, s["p0"]], [s["t1"], s["p1"]], [s["t2"], s["p2"]]])])
fam("po04", PO, ["When should we place the next order of {item}? On hand {on_hand}, demand {d}/day, lead time {L} days, safety stock {ss}.",
                 "How many days until I need to reorder {item}: {on_hand} in stock, {d} units/day, {L}-day lead time, {ss} safety stock?"],
    lambda s: [calc("days_until_reorder", on_hand=s["on_hand"], avg_daily_demand=s["d"], lead_time_days=s["L"], safety_stock=s["ss"])])
fam("po05", PO, ["Forecast demand for {item}{at_store} {h} and tell me when to place the next order with {on_hand} on hand, {L} days lead time and {ss} safety stock.",
                 "Predict {item} demand{at_store} {h}, then work out how soon I must reorder ({on_hand} units, lead time {L} days, safety stock {ss})."],
    lambda s: [fc(s), calc("days_until_reorder", on_hand=s["on_hand"], avg_daily_demand=MD, lead_time_days=s["L"], safety_stock=s["ss"])])
fam("po06", PO, ["Is it cheaper to order {item} weekly or monthly? Annual demand {D}, ordering cost ${S}, holding cost ${H}/unit/year.",
                 "Weekly versus monthly ordering for {item}: D={D}, S=${S}, H=${H}. Which costs less?"],
    lambda s: [py("compare_order_policies", annual_demand=s["D"], order_cost=s["S"], holding_cost_per_unit=s["H"], orders_per_year_options=[52, 12])])
fam("po07", PO, ["Find the current price of {prod} per unit and calculate how many I can buy with ${budget}.",
                 "Search for a typical unit price of {prod} and tell me how many units ${budget} gets me."],
    lambda s: [ws(f"{s['prod']} price per unit"),
               calc("budget_coverage", budget=s["budget"], unit_cost="$extract(step1.results, unit_price)", avg_daily_demand=None)])
fam("po08", PO, ["Forecast demand for {item}{at_store} {h}, calculate the EOQ using ordering cost ${S} and holding cost ${H}/unit/year (annualize the forecast), and simulate stockout risk with {on_hand} units on hand, {L}-day lead time and demand std {sd}.",
                 "Plan procurement for {item}{at_store}: forecast {h}, compute EOQ (S=${S}, H=${H}, annualized forecast) and run a stockout simulation ({on_hand} units, lead time {L}, std {sd})."],
    lambda s: [fc(s), calc("eoq", annual_demand=MD + " * 365", order_cost=s["S"], holding_cost_per_unit=s["H"]),
               py("monte_carlo_stockout", on_hand=s["on_hand"], mean_daily_demand=MD, std_daily_demand=s["sd"], lead_time_days=s["L"])])
fam("po09", PO, ["How many orders per year does the EOQ imply? Annual demand {D}, ordering cost ${S}, holding cost ${H} per unit-year.",
                 "Compute the EOQ and the resulting number of orders per year (D={D}, S=${S}, H=${H})."],
    lambda s: [calc("eoq", annual_demand=s["D"], order_cost=s["S"], holding_cost_per_unit=s["H"]),
               calc("order_frequency", annual_demand=s["D"], order_quantity="$step1.eoq")])
fam("po10", PO, ["Calculate safety stock for a {sl_pct}% service level (std dev {sd}/day, lead time {L} days) and then the reorder point at {d} units/day.",
                 "First find safety stock ({sl_pct}% service, demand std {sd}, {L} days), then the reorder point with average demand {d}/day."],
    lambda s: [calc("safety_stock", service_level=s["sl"], std_daily_demand=s["sd"], lead_time_days=s["L"]),
               calc("reorder_point", avg_daily_demand=s["d"], lead_time_days=s["L"], safety_stock="$step1.safety_stock")])

# ---- families whose gold calls have incomplete args in the template (needed later steps) are completed here
# (df10 on_hand/lead_time and po07 avg_daily_demand are intentionally absent from text -> drop null keys)


def clean_args(call):
    call["args"] = _strip(call["args"])
    return call


def _strip(o):
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if v is not None or k in ("store_id",)}
    if isinstance(o, list):
        return [_strip(v) for v in o]
    return o


# ------------------------------------------------------------------ augmentation
PREFIX = ["", "", "", "Hey, ", "Quick question: ", "Can you help me with this? "]
SUFFIX = ["", "", "", "", " Thanks!", " Be brief."]


KEEP_CAP = {"Supplier", "Product", "I", "We", "Our"}


def augment(text, rng):
    pre = rng.choice(PREFIX)
    if pre in ("Hey, ", "Please ") and text.split()[0] not in KEEP_CAP and text.split()[0].isalpha():
        text = text[0].lower() + text[1:]
    text = pre + text + rng.choice(SUFFIX)
    if rng.random() < 0.12:
        text = text.rstrip("?.!")
    return text


def skeleton(text):
    t = re.sub(r"[A-Z]+_\d_\d+", "<ID>", text)
    t = re.sub(r"\$?\d[\d,\.]*%?", "<N>", t)
    return t.lower().strip()


# ------------------------------------------------------------------ build + split
def build(per_family, seed):
    rng = random.Random(seed)
    rows, seen = [], set()
    for fid, intent, variants, fn in F:
        made, tries = 0, 0
        while made < per_family and tries < per_family * 20:
            tries += 1
            s = sample_slots(rng)
            vi = rng.randrange(len(variants))
            text = augment(variants[vi].format(**s), rng)
            if text in seen:
                continue
            calls = [clean_args(c) for c in fn(s)]
            tools = []
            for c in calls:
                lab = F2L[c["tool"]]
                if lab not in tools:
                    tools.append(lab)
            seen.add(text)
            rows.append({"query": text, "intent": intent, "tools": sorted(tools), "tool_calls": calls,
                         "family_id": fid, "variant_id": vi, "source": "template"})
            made += 1
    for i, r in enumerate(rows):
        r["id"] = f"nlp_{i:05d}"
    return rows


def split_families(rows, seed, n_val=2, n_test=2, trials=20000):
    """Family-level split (no paraphrase leakage), stratified by intent: per intent n_val/n_test families
    go to val/test, rest to train. A seeded random search picks the assignment that (a) covers every tool
    label and many distinct tool-sets in val and test, and (b) never puts a tool-set in val/test that is
    absent from train."""
    rng = random.Random(seed)
    fams = {}
    for r in rows:
        fams.setdefault(r["family_id"], (r["intent"], tuple(r["tools"])))
    by_intent = defaultdict(list)
    for f, (intent, ts) in fams.items():
        by_intent[intent].append(f)

    def score(assign):
        sc = 0
        train_ts = {fams[f][1] for f, s in assign.items() if s == "train"}
        for sp in ("val", "test"):
            fl = [f for f, s in assign.items() if s == sp]
            labels = {lab for f in fl for lab in fams[f][1]}
            tsets = {fams[f][1] for f in fl}
            multi = sum(len(fams[f][1]) > 1 for f in fl)
            sc += 5 * len(labels) + len(tsets) + 2 * min(multi, 4)
            sc -= 10 * len([ts for ts in tsets if ts not in train_ts])
        return sc

    best, best_sc = None, -10**9
    for _ in range(trials):
        assign = {}
        for intent, fl in by_intent.items():
            sh = fl[:]
            rng.shuffle(sh)
            for i, f in enumerate(sh):
                assign[f] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
        sc = score(assign)
        if sc > best_sc:
            best, best_sc = assign, sc
    for r in rows:
        r["split"] = best[r["family_id"]]
    return rows


def load_config():
    """Read configs/nlp_dataset.yaml (flat key: value). Uses PyYAML if installed, else a tiny parser."""
    path = ROOT / "configs/nlp_dataset.yaml"
    try:
        import yaml
        return yaml.safe_load(path.read_text())
    except ImportError:
        cfg = {}
        for line in path.read_text().splitlines():
            line = line.split("#")[0].strip()
            if ":" in line:
                k, v = line.split(":", 1)
                cfg[k.strip()] = int(v.strip())
        return cfg


def main(argv=None):
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=cfg["per_family"])
    ap.add_argument("--seed", type=int, default=cfg["seed"])
    a = ap.parse_args(argv)
    rows = build(a.per_family, a.seed)
    rows = split_families(rows, a.seed, cfg["val_families_per_intent"], cfg["test_families_per_intent"],
                          cfg["split_search_trials"])
    out = ROOT / "data/nlp"
    (out / "splits").mkdir(parents=True, exist_ok=True)
    with open(out / "nlp_queries.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    for sp in ["train", "val", "test"]:
        with open(out / "splits" / f"{sp}.jsonl", "w") as f:
            for r in rows:
                if r["split"] == sp:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows from {len(F)} families")


if __name__ == "__main__":
    main()
