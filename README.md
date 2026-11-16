# SupplyIQ — Intelligent Supply Chain Analytics & NLP Agent

NLP final project. A natural-language supply chain assistant built from:

1. **XGBoost demand forecaster** trained on M5 (shared tool)
2. **Experiment 1:** three trained NLP routers (TF-IDF+SVM, TextCNN, RoBERTa). Each predicts the
   *primary intent* (multi-class) and the *required tools* (multi-label).
3. **Experiment 2:** three tool-calling agents (ReAct, Plan-and-Execute, ReAct+Self-Verification).
   All three use the best router from Exp. 1, the same LLM and the same tools.

## Repository layout

```
SupplyIQ/
├── configs/                 # YAML hyperparameters per experiment (one file per model)
├── data/
│   ├── raw/m5/              # M5 CSVs from Kaggle            (gitignored)
│   ├── processed/           # parquet features/splits        (gitignored, regenerable)
│   ├── reference/           # small committed tables: product catalog, documented assumptions
│   └── nlp/                 # labeled queries (committed). splits/ = frozen train/val/test
├── docs/
│   ├── annotation_guidelines.md   # labeling rules for the NLP dataset (Step 2)
│   ├── decisions.md               # design-decision log (why we chose X)
│   └── weekly_reports/
├── notebooks/               # exploration only; name as NN_initials_topic.ipynb
├── src/supplyiq/            # importable package: all real logic lives here
│   ├── config.py            # paths + global seed (single source of truth)
│   ├── data/                # M5 loading, feature engineering, NLP dataset + splitting
│   ├── forecasting/         # baselines, XGBoost train/predict
│   ├── nlp/                 # svm.py, textcnn.py, roberta.py + shared metrics
│   ├── tools/               # forecast / calculator / python sandbox / web search + registry
│   ├── agents/              # react.py, plan_execute.py, react_verify.py
│   ├── eval/                # NLP + agent evaluation harnesses
│   └── api/                 # FastAPI backend (final integration)
├── scripts/                 # thin CLI entry points (python scripts/xxx.py)
├── tests/                   # pytest; tools + calculators must be unit-tested
├── models/                  # trained artifacts (gitignored, share via Drive)
├── results/                 # committed metrics (json/csv) + figures used in the report
└── app/                     # optional Streamlit demo
```

Rule of thumb: **notebooks explore, `src/` implements.** Anything another member depends on
must live in `src/` and be importable, not copy-pasted from a notebook.

## Setup (each member, once)

Requires **Python 3.11 or 3.12**. Do not use 3.13, because not all ML wheels support it yet.

**Windows (PowerShell):**
```powershell
cd SupplyIQ
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 1) PyTorch FIRST. Pick ONE line:
pip install torch --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install torch --index-url https://download.pytorch.org/whl/cpu     # no GPU

# 2) Everything else + our package in editable mode
pip install -r requirements.txt
pip install -e .

# 3) Verify
python scripts/check_env.py
```

**macOS / Linux:** same steps, but use `python3.12 -m venv .venv`, `source .venv/bin/activate`,
and plain `pip install torch` (on Apple Silicon this gives you the MPS build).

### Kaggle credentials (needed to download M5)
1. Open https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules and click **"I Understand and Accept"**.
   If you skip this, the download returns 403.
2. Kaggle → Settings → API → **Create New Token**. This downloads `kaggle.json`.
3. Put it at `C:\Users\<you>\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (mac/Linux, then `chmod 600`).

## Team workflow

| Member | Exp. 1 (NLP model) | Exp. 2 (Agent) |
|---|---|---|
| Member 1 | TF-IDF + SVM | ReAct |
| Member 2 | TextCNN | Plan-and-Execute |
| Member 3 | RoBERTa (fine-tune) | ReAct + Self-Verification |
| Shared | M5 prep, XGBoost, NLP dataset, tools, eval harness, integration, report | |

Full rules are in **[CONTRIBUTING.md](CONTRIBUTING.md)**. The short version:

- `main` is protected. Work on `<firstname>/dev`, merge via PR with 1 approval + green CI.
- **Frozen artifacts:** after `data/nlp/splits/` and the evaluation metrics code are merged, nobody edits
  them without team agreement (CI enforces this via the `frozen-change-approved` PR label).
- **Never look at test-set scores while tuning.** Use validation only. Test is run once per model at the end.
- Record every non-obvious choice in `docs/decisions.md`.
