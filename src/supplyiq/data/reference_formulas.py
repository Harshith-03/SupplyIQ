"""Reference implementations of inventory formulas and analyses.

Single source of truth: used (1) to compute ground-truth answers for the agent
benchmark and (2) later as the backend of the calculator / python tools, so the
benchmark and the tools can never disagree.
"""
import math
from statistics import NormalDist


def z_from_service_level(service_level: float) -> float:
    if not 0.5 < service_level < 1:
        raise ValueError("service_level must be in (0.5, 1)")
    return NormalDist().inv_cdf(service_level)


def safety_stock(service_level, std_daily_demand, lead_time_days):
    _pos(std_daily_demand=std_daily_demand, lead_time_days=lead_time_days)
    return z_from_service_level(service_level) * std_daily_demand * math.sqrt(lead_time_days)


def reorder_point(avg_daily_demand, lead_time_days, safety_stock=0.0):
    _nonneg(avg_daily_demand=avg_daily_demand, lead_time_days=lead_time_days, safety_stock=safety_stock)
    return avg_daily_demand * lead_time_days + safety_stock


def eoq(annual_demand, order_cost, holding_cost_per_unit):
    _pos(annual_demand=annual_demand, order_cost=order_cost, holding_cost_per_unit=holding_cost_per_unit)
    return math.sqrt(2 * annual_demand * order_cost / holding_cost_per_unit)


def days_of_supply(on_hand, avg_daily_demand):
    _nonneg(on_hand=on_hand)
    if avg_daily_demand <= 0:
        raise ValueError("avg_daily_demand must be > 0 (days of supply undefined)")
    return on_hand / avg_daily_demand


def order_frequency(annual_demand, order_quantity):
    _pos(annual_demand=annual_demand, order_quantity=order_quantity)
    return annual_demand / order_quantity


def inventory_turnover(cogs, average_inventory_value):
    _pos(cogs=cogs, average_inventory_value=average_inventory_value)
    return cogs / average_inventory_value


def order_quantity(target_demand, on_hand, safety_stock=0.0):
    _nonneg(target_demand=target_demand, on_hand=on_hand, safety_stock=safety_stock)
    return max(0.0, target_demand + safety_stock - on_hand)


def days_until_reorder(on_hand, avg_daily_demand, lead_time_days, safety_stock=0.0):
    rop = reorder_point(avg_daily_demand, lead_time_days, safety_stock)
    if avg_daily_demand <= 0:
        raise ValueError("avg_daily_demand must be > 0")
    return max(0.0, (on_hand - rop) / avg_daily_demand)


def annual_holding_cost(average_inventory_units, unit_cost, holding_rate):
    _nonneg(average_inventory_units=average_inventory_units, unit_cost=unit_cost, holding_rate=holding_rate)
    return average_inventory_units * unit_cost * holding_rate


def period_holding_cost(average_inventory_units, unit_cost, holding_rate, days):
    return annual_holding_cost(average_inventory_units, unit_cost, holding_rate) * days / 365.0


def total_annual_cost(annual_demand, order_quantity, order_cost, holding_cost_per_unit, unit_cost=0.0):
    _pos(annual_demand=annual_demand, order_quantity=order_quantity)
    return (annual_demand * unit_cost
            + annual_demand / order_quantity * order_cost
            + order_quantity / 2 * holding_cost_per_unit)


def stockout_cost(units_short, unit_margin):
    _nonneg(units_short=units_short, unit_margin=unit_margin)
    return units_short * unit_margin


def landed_cost(unit_price, quantity, shipping_cost=0.0):
    _nonneg(unit_price=unit_price, quantity=quantity, shipping_cost=shipping_cost)
    total = unit_price * quantity + shipping_cost
    return {"total_cost": total, "cost_per_unit": total / quantity if quantity else float("nan")}


def budget_coverage(budget, unit_cost, avg_daily_demand=None):
    _pos(budget=budget, unit_cost=unit_cost)
    units = math.floor(budget / unit_cost)
    days = None
    if avg_daily_demand is not None:
        _pos(avg_daily_demand=avg_daily_demand)
        days = units / avg_daily_demand
    return {"units": units, "days_covered": days}


def price_change_impact(annual_spend, pct_change):
    delta = annual_spend * pct_change / 100.0
    return {"delta": delta, "new_spend": annual_spend + delta}


# ---------------- python-tool style analyses (deterministic ones) ----------------

def simulate_delay(on_hand, daily_demand, lead_time_days, delay_days):
    """Deterministic constant-demand model. A shipment is due after lead_time_days,
    but arrives lead_time_days + delay_days instead."""
    _pos(daily_demand=daily_demand)
    cover = on_hand / daily_demand
    normal_short = max(0.0, lead_time_days - cover) * daily_demand
    delayed_short = max(0.0, lead_time_days + delay_days - cover) * daily_demand
    return {
        "days_of_cover": cover,
        "stockout_day": cover if cover < lead_time_days + delay_days else None,
        "shortage_units_normal": normal_short,
        "shortage_units_delayed": delayed_short,
        "extra_shortage_units": delayed_short - normal_short,
    }


def compare_suppliers(suppliers, criterion, quantity=None, weights=None):
    """suppliers: list of dicts with name, unit_price, lead_time_days, optional shipping_cost, on_time_rate.
    criterion: lowest_unit_price | lowest_total_cost | weighted | highest_on_time_rate
    weighted score = w_price*(min_price/price) + w_lead*(min_lead/lead); highest wins."""
    if criterion == "lowest_unit_price":
        key = {s["name"]: s["unit_price"] for s in suppliers}
        best = min(key, key=key.get)
    elif criterion == "lowest_total_cost":
        key = {s["name"]: s["unit_price"] * quantity + s.get("shipping_cost", 0) for s in suppliers}
        best = min(key, key=key.get)
    elif criterion == "weighted":
        mp = min(s["unit_price"] for s in suppliers)
        ml = min(s["lead_time_days"] for s in suppliers)
        key = {s["name"]: weights["price"] * mp / s["unit_price"] + weights["lead_time"] * ml / s["lead_time_days"]
               for s in suppliers}
        best = max(key, key=key.get)
    elif criterion == "highest_on_time_rate":
        key = {s["name"]: s["on_time_rate"] for s in suppliers}
        best = max(key, key=key.get)
    else:
        raise ValueError(criterion)
    return {"best": best, "scores": key}


def best_price_break(annual_demand, order_cost, holding_rate, breaks):
    """breaks: list of (min_qty, unit_price) ascending. Holding cost = price * holding_rate."""
    breaks = sorted(breaks)
    out = []
    for i, (min_q, price) in enumerate(breaks):
        h = price * holding_rate
        q = max(eoq(annual_demand, order_cost, h), min_q)
        if i + 1 < len(breaks):
            q = min(q, breaks[i + 1][0] - 1)
        total = total_annual_cost(annual_demand, q, order_cost, h, price)
        out.append({"min_qty": min_q, "unit_price": price, "order_qty": q, "total_cost": total})
    best = min(out, key=lambda r: r["total_cost"])
    return {"best": best, "all": out}


def hhi_concentration(shares_pct):
    if abs(sum(shares_pct) - 100) > 1e-6:
        raise ValueError("shares must sum to 100")
    return sum(s * s for s in shares_pct)  # 0..10000 scale


def stockout_probability_normal(on_hand, mean_daily_demand, std_daily_demand, lead_time_days):
    """P(lead-time demand > on_hand), demand ~ Normal (independent days)."""
    mu = mean_daily_demand * lead_time_days
    sd = std_daily_demand * math.sqrt(lead_time_days)
    return 1 - NormalDist(mu, sd).cdf(on_hand)


def scenario_demand_shock(daily_demand, shock_pct, on_hand, lead_time_days):
    shocked = daily_demand * (1 + shock_pct / 100.0)
    return {
        "shocked_daily_demand": shocked,
        "days_of_cover": on_hand / shocked,
        "shortage_units": max(0.0, shocked * lead_time_days - on_hand),
    }


def compare_order_policies(annual_demand, order_cost, holding_cost_per_unit, orders_per_year_options):
    res = {}
    for n in orders_per_year_options:
        q = annual_demand / n
        res[n] = total_annual_cost(annual_demand, q, order_cost, holding_cost_per_unit)
    best = min(res, key=res.get)
    return {"costs": res, "best_orders_per_year": best}


def _pos(**kw):
    for k, v in kw.items():
        if v is None or v <= 0:
            raise ValueError(f"{k} must be > 0 (got {v})")


def _nonneg(**kw):
    for k, v in kw.items():
        if v is None or v < 0:
            raise ValueError(f"{k} must be >= 0 (got {v})")
