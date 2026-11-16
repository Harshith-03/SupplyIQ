## What does this PR do?
<!-- 1-3 sentences. Link the task / week. -->

## Type
- [ ] Data / preprocessing
- [ ] Model (SVM / TextCNN / RoBERTa / XGBoost)
- [ ] Tools / agents
- [ ] Evaluation
- [ ] Docs / infra

## How was it tested?
<!-- Commands run + output. "It runs" is not enough for model code; include val metrics. -->

## Checklist
- [ ] `ruff check .` and `pytest -q` pass locally
- [ ] No raw data, model weights, `.env` or keys committed
- [ ] Tuning used the **validation** split only. No test-set numbers were looked at.
- [ ] Any reported numbers were actually produced (commit + config noted)
- [ ] Non-obvious choices logged in `docs/decisions.md`
- [ ] Touches frozen files? If so, team agreed and the `frozen-change-approved` label is added
