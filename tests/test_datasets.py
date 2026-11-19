"""Tests for Dataset 4 (NLP queries) and Dataset 5 (agent benchmark). Reads the committed data files."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from supplyiq.data import dataset_validation as V
from supplyiq.data import reference_formulas as R


def _load(p):
    return [json.loads(line) for line in open(ROOT / p)]


def test_full_validation_has_no_errors():
    errors, _warnings, _report = V.validate()
    assert errors == []


def test_splits_files_match_master_file():
    master = _load("data/nlp/nlp_queries.jsonl")
    for sp in ("train", "val", "test"):
        rows = _load(f"data/nlp/splits/{sp}.jsonl")
        assert rows and all(r["split"] == sp for r in rows)
        assert {r["id"] for r in rows} == {r["id"] for r in master if r["split"] == sp}


def test_no_family_spans_splits():
    fam = defaultdict(set)
    for r in _load("data/nlp/nlp_queries.jsonl"):
        fam[r["family_id"]].add(r["split"])
    assert all(len(s) == 1 for s in fam.values())


def test_every_intent_in_every_split():
    for sp in ("train", "val", "test"):
        assert len({r["intent"] for r in _load(f"data/nlp/splits/{sp}.jsonl")}) == 6


def test_benchmark_has_all_tiers_and_criteria():
    ev = _load("data/eval/agent_benchmark.jsonl")
    assert {t["tier"] for t in ev} == {"single_tool", "multi_tool", "multi_step", "edge_case"}
    assert all(t["completion_criteria"] for t in ev)


def test_reference_formulas_known_values():
    assert abs(R.eoq(24000, 75, 3.2) - 1060.66) < 0.01
    assert abs(R.safety_stock(0.95, 18, 16) - 118.43) < 0.01
    assert R.reorder_point(120, 9, 150) == 1230
    assert R.simulate_delay(900, 60, 10, 7)["shortage_units_delayed"] == 120
