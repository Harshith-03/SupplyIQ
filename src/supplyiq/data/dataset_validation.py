"""Validate Dataset 4 (NLP) and Dataset 5 (agent benchmark).  CLI: python scripts/validate_datasets.py
Exit code 1 if any hard error is found."""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from supplyiq.data import reference_formulas as R
from supplyiq.data.nlp_dataset import skeleton

SCHEMA = json.loads((ROOT / "data/nlp/label_schema.json").read_text())
INTENTS, LABELS = SCHEMA["intents"], SCHEMA["tool_labels"]
F2L = {v: k for k, v in SCHEMA["tool_label_to_function"].items()}
CALC_OPS = SCHEMA["tool_contracts"]["inventory_calculator"]["operations"]
PY_TASKS = SCHEMA["tool_contracts"]["python_analysis"]["tasks"]
errors, warns = [], []


def load(p):
    return [json.loads(line) for line in open(p)]


# ------- replay engine: executes gold calls with reference formulas (dummy forecast) -------
DUMMY = {"mean_daily_demand": 50.0, "total_demand": 1400.0, "daily_forecast": [50.0] * 7}


def resolve(v, outs):
    if isinstance(v, str) and v.startswith("$step"):
        m = re.match(r"\$step(\d+)\.(\w+)(.*)", v)
        base = outs[int(m.group(1)) - 1][m.group(2)]
        return eval(f"{base!r}{m.group(3)}") if m.group(3).strip() else base
    if isinstance(v, dict):
        return {k: resolve(x, outs) for k, x in v.items()}
    if isinstance(v, list):
        return [resolve(x, outs) for x in v]
    return v


def run_call(call, outs):
    t, a = call["tool"], call["args"]
    if t == "forecast_demand":
        return dict(DUMMY)
    if t == "web_search":
        return {"results": "<web>"}
    p = a.get("params", {})
    if any(isinstance(x, str) and "$extract" in x for x in json.dumps(p).split('"')):
        return {"_skipped": "llm-extracted"}
    p = resolve(p, outs)
    if t == "inventory_calculator":
        op = a["operation"]
        fn = getattr(R, op)
        r = fn(**p)
        return r if isinstance(r, dict) else {op: r}
    task = a["task"]
    if task == "simulate_delay":
        return R.simulate_delay(**p)
    if task == "compare_suppliers":
        return R.compare_suppliers(**p)
    if task == "best_price_break":
        return R.best_price_break(p["annual_demand"], p["order_cost"], p["holding_rate"], [tuple(b) for b in p["breaks"]])
    if task == "hhi_concentration":
        return {"hhi": R.hhi_concentration(p["shares_pct"])}
    if task == "scenario_demand_shock":
        if None in (p.get("on_hand"), p.get("lead_time_days")) or "on_hand" not in p:
            return {"_skipped": "partial-args by design"}
        return R.scenario_demand_shock(**p)
    if task == "compare_order_policies":
        return R.compare_order_policies(p["annual_demand"], p["order_cost"], p["holding_cost_per_unit"], p["orders_per_year_options"])
    return {"_skipped": "no reference impl (needs M5 data / simulation)"}


def check_calls(calls, where):
    for i, c in enumerate(calls):
        if c["tool"] not in F2L:
            errors.append(f"{where}: unknown tool {c['tool']}")
        if c["tool"] == "inventory_calculator" and c["args"]["operation"] not in CALC_OPS:
            errors.append(f"{where}: unknown calc op {c['args']['operation']}")
        if c["tool"] == "python_analysis" and c["args"]["task"] not in PY_TASKS:
            errors.append(f"{where}: unknown python task {c['args']['task']}")
        if c["tool"] == "inventory_calculator":
            need = CALC_OPS.get(c["args"]["operation"], [])
            have = set(c["args"]["params"])
            miss = [k for k in need if k not in have and k not in ("safety_stock", "unit_cost", "shipping_cost", "avg_daily_demand")]
            if miss:
                errors.append(f"{where}: calc {c['args']['operation']} missing params {miss}")
        for m in re.finditer(r"\$step(\d+)", json.dumps(c["args"])):
            if int(m.group(1)) > i:
                errors.append(f"{where}: call {i+1} references later/same step {m.group(1)}")


def replay(calls, where, stats):
    outs = []
    try:
        for c in calls:
            o = run_call(c, outs)
            outs.append(o)
            stats["skipped" if "_skipped" in o else "executed"] += 1
    except Exception as e:  # noqa
        errors.append(f"{where}: replay failed: {type(e).__name__}: {e}")


def validate(m5_items_path=None):
    """Returns (errors, warnings, report_lines)."""
    errors.clear()
    warns.clear()
    nlp = load(ROOT / "data/nlp/nlp_queries.jsonl")
    ev = load(ROOT / "data/eval/agent_benchmark.jsonl")
    m5 = set(Path(m5_items_path).read_text().split()) if m5_items_path else None

    # ---------- Dataset 4
    stats = Counter()
    for r in nlp:
        w = r["id"]
        if r["intent"] not in INTENTS:
            errors.append(f"{w}: bad intent")
        if not set(r["tools"]) <= set(LABELS) or not r["tools"]:
            errors.append(f"{w}: bad tools {r['tools']}")
        if sorted({F2L[c['tool']] for c in r["tool_calls"]}) != sorted(r["tools"]):
            errors.append(f"{w}: tools label != tool_calls")
        check_calls(r["tool_calls"], w)
        replay(r["tool_calls"], w, stats)
        if m5:
            for c in r["tool_calls"]:
                iid = c["args"].get("item_id") or c["args"].get("params", {}).get("item_id")
                if iid and iid not in m5:
                    errors.append(f"{w}: item_id {iid} not in M5 subset")
    # duplicates & leakage
    if len({r["query"] for r in nlp}) != len(nlp):
        errors.append("exact duplicate queries")
    fam_splits = defaultdict(set)
    for r in nlp:
        fam_splits[r["family_id"]].add(r["split"])
    for f, s in fam_splits.items():
        if len(s) > 1:
            errors.append(f"family {f} spans splits {s}")
    sk = defaultdict(set)
    for r in nlp:
        sk[skeleton(r["query"])].add(r["split"])
    leaks = [k for k, s in sk.items() if len(s) > 1]
    if leaks:
        warns.append(f"{len(leaks)} number-masked skeletons appear in >1 split (near-duplicate leak), e.g. {leaks[0][:80]}")
    for sp in ("val", "test"):
        tr = {tuple(r["tools"]) for r in nlp if r["split"] == "train"}
        miss = {tuple(r["tools"]) for r in nlp if r["split"] == sp} - tr
        if miss:
            errors.append(f"{sp} has tool-sets unseen in train: {miss}")
    for sp in ("train", "val", "test"):
        cnt = Counter(tuple(r["tools"]) for r in nlp if r["split"] == sp)
        for ts, n in cnt.items():
            if n < 50 and sp != "train":
                warns.append(f"{sp}: tool-set {ts} has only {n} examples (macro-F1 per label is noisy)")
    for ts in {tuple(r["tools"]) for r in nlp}:
        n_f = len({r["family_id"] for r in nlp if tuple(r["tools"]) == ts})
        if n_f == 1:
            warns.append(f"tool-set {ts} comes from a single family only (not in val/test) - add hand-written examples")

    # ---------- Dataset 5
    stats5 = Counter()
    nlp_sk = {skeleton(r["query"]) for r in nlp}
    for t in ev:
        w = t["id"]
        check_calls(t["expected_tool_calls"], w)
        if t["grading"].get("expected_behavior") in ("report_invalid_input", "handle_division_by_zero"):
            try:
                run_call(t["expected_tool_calls"][0], [])
                errors.append(f"{w}: expected the reference tool to reject this input, but it did not")
            except ValueError:
                stats5["rejected_as_designed"] += 1
        else:
            replay(t["expected_tool_calls"], w, stats5)
        if skeleton(t["query"]) in nlp_sk:
            errors.append(f"{w}: eval query skeleton also in NLP dataset (leak)")
        if not t["completion_criteria"]:
            errors.append(f"{w}: no completion criteria")
        if t["grading"]["type"] == "numeric" and "expected" not in t["grading"] and "expected_text" not in t["grading"]:
            errors.append(f"{w}: numeric task without expected values")
        if m5:
            for c in t["expected_tool_calls"]:
                iid = c["args"].get("item_id")
                if iid and iid not in m5 and not iid.startswith("XYZ"):
                    errors.append(f"{w}: item_id {iid} not in M5 subset")

    report = [
        f"Dataset 4: {len(nlp)} rows | splits {dict(Counter(r['split'] for r in nlp))}",
        f"  intents: {dict(Counter(r['intent'] for r in nlp))}",
        f"  #tools per query: {dict(sorted(Counter(len(r['tools']) for r in nlp).items()))}",
        (
            f"  gold-call replay: executed={stats['executed']} skipped={stats['skipped']} "
            "(skips need M5 data / LLM extraction)"
        ),
        f"Dataset 5: {len(ev)} tasks | tiers {dict(Counter(t['tier'] for t in ev))}",
        (
            f"  gold-call replay: executed={stats5['executed']} skipped={stats5['skipped']} "
            f"rejected_as_designed={stats5['rejected_as_designed']}"
        ),
    ]
    return list(errors), list(warns), report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m5-items", help="optional text file with one M5 item_id per line")
    a = ap.parse_args()
    errs, wrn, report = validate(a.m5_items)
    print("\n".join(report))
    for w in wrn:
        print("WARN :", w)
    for e in errs[:40]:
        print("ERROR:", e)
    print("RESULT:", "FAIL" if errs else "PASS", f"({len(errs)} errors, {len(wrn)} warnings)")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
