# Contributing to SupplyIQ

Every team member, and every AI assistant helping a member, follows these rules.

## 1. Branches
- `main` is protected. Nobody pushes to it directly, and every change lands through a Pull Request.
- Each member works on their own branch named `<firstname>/dev` (e.g. `harshith/dev`).
  For a bigger self-contained task you may also create `<firstname>/<topic>` (e.g. `harshith/svm-baseline`).
- Keep your branch current with `main` at least before opening a PR:
  ```bash
  git fetch origin
  git merge origin/main        # resolve conflicts, re-run tests, push
  ```

## 2. Daily loop
```bash
git checkout <firstname>/dev
git pull                      # your own branch
git merge origin/main         # pick up teammates' merged work (after git fetch)
# ... work ...
ruff check .                  # lint (same as CI)
pytest -q                     # tests (same as CI)
git add <specific files>      # never `git add .` blindly, check `git status` first
git commit -m "nlp: add tfidf+svm training script"
git push
```
Commit messages start with an area prefix: `data:`, `forecast:`, `nlp:`, `tools:`, `agents:`, `eval:`, `docs:`, `ci:`.

## 3. Pull Requests
- Open the PR from your branch into `main` and fill in the template.
- Merging requires **1 approval from another member** and **green CI**.
- CI (`.github/workflows/ci.yml`) runs four checks:
  | Check | Fails when |
  |---|---|
  | `lint` | `ruff check .` reports problems |
  | `test` | any `pytest` test fails |
  | `repo-hygiene` | the PR adds `.env`, `kaggle.json`, raw/processed data, model weights, or any file > 5 MB |
  | `frozen-files` | the PR changes a path in `.github/frozen-paths.txt` without the `frozen-change-approved` label |

## 4. Research-integrity rules (non-negotiable)
1. **Same data for everyone.** All three NLP models use the identical frozen splits in `data/nlp/splits/`.
   All three agents use the same router, LLM, tools and benchmark.
2. **Validation for tuning, test exactly once.** Never compute or look at test-set scores while choosing
   hyperparameters or models. The test set is evaluated once per model at the end.
3. **Shared metrics code only.** Report NLP/agent metrics with the functions in `src/supplyiq/eval/`.
   Don't write private metric code, because it makes numbers incomparable.
4. **No invented results.** Every number in a report or PR must come from a run you can point to
   (commit + config + command).
5. **No fabricated supply-chain facts.** Prices, lead times, stock levels and supplier data must come
   from the dataset, a tool call, or a documented assumption in `data/reference/`.
6. **Log decisions** in `docs/decisions.md` (anything that affects results or fairness).

## 5. What goes where
- Reusable code goes in `src/supplyiq/` (importable as `supplyiq.*`). Notebooks are for exploration only.
- CLI entry points go in `scripts/`. Hyperparameters go in `configs/*.yaml`, never hardcoded.
- Do not commit: `data/raw/`, `data/processed/`, `models/`, `.env`.
  Share trained weights via the team Drive folder and record the link in the PR.
- Do commit: `data/nlp/` (labeled queries + splits), `data/reference/`, `results/` (metrics + figures).

## 6. Secrets
Copy `.env.example` to `.env` and put API keys there. Never paste keys into code, notebooks, issues or PRs.
