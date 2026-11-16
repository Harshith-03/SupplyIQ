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
