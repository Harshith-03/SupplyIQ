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
