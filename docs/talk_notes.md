# 30-minute talk notes

Before presenting: install Python 3.11 and Git; run `python scripts/run_all.py`. Open both notebooks and run each once so their kernels hold the variables. Keep saved outputs visible. Use these notes outside the notebooks. Numbers below come from the executed notebooks; full execution took 10.42s and 39.74s on the tested Linux machine, not a measured Windows laptop.

| Time | Section and what to run | What to say |
|---|---|---|
| 0–2 min | ML 1: saved baseline table/chart | “Approving everything looks accurate because fraud is rare. On validation it is 98.79% accurate and catches zero fraud; precision measures useful alerts, recall measures fraud caught.” |
| 2–3 | ML 2: saved head/types | “This is one synthetic table with 50,150 raw rows, including duplicates. All spending patterns and costs are assumptions.” |
| 3–5 | ML 3: run cleaning summaries live | “We remove duplicate and invalid rows, then standardise categories. The median comes only from training data; future rows cannot influence it.” |
| 5–8 | ML 4: use the five saved figures | “These figures use the training period only. Night, unusually large amounts and long distances help describe our simulated risk rule, but fraud and legitimate payments overlap.” |
| 8–9 | ML 5: show saved feature list | “Relative amount compares a payment with that customer's usual spend. Each of these four features uses only the current row.” |
| 9–10 | ML 6: show timeline | “We train on the first 60 days, select on the next 15 and test on the last 15. A random split can mix behaviour from the past and future.” |
| 10–12 | ML 7: rerun fit cell live | “The same preprocessing structure supports three models. Only logistic regression needs numeric scaling; all fitted preprocessing comes from training rows.” |
| 12–14 | ML 8: show saved comparison/PR curves | “LightGBM's validation PR-AUC is 0.392. Random Forest catches more fraud at its chosen threshold, but costs more under our assumptions.” |
| 14–16 | ML 9: run selection cell live | “We choose LightGBM at 0.67, with recall 69.61% and INR 14.90 per transaction. The threshold must stay within 45 alerts on every validation day.” |
| 16–17 | ML 10: use saved final test; do not rerun live | “The frozen choice reaches 77.14% recall, 20.25% precision and 0.459 PR-AUC on later transactions. These are synthetic results, not evidence of production performance.” |
| 17–19 | MLOps 1: run Git demo live | “A branch lets us make a small config change and merge it. A commit identifies exactly which files we used; this demo changes only a note.” |
| 19–22 | MLOps 2: rerun tracking/UI cells live | “An experiment is a saved record of settings, results and a fitted model. These three runs use local SQLite and files; the Git SHA and config hash tell us where they came from.” |
| 22–25 | MLOps 3: run registration, promotion and rollback live | “A registered version is fixed; an alias is a movable name. We move champion to the challenger and back without retraining or deploying.” |
| 25–28 | MLOps 4: run pytest and paired gates live | “Tests check code behaviour; a quality gate checks model acceptability. The real model exits 0, while the baseline exits 1 for low PR-AUC, zero recall and excessive cost.” |
| 28–30 | MLOps 5: show saved workflow/local CI and diagram | “A future GitHub push runs these same checks. Fresh CI compares against a bootstrap baseline; choosing a registry alias is still a separate decision.” |

**Two audience questions**

1. “Would you trust a model that approves every transaction?” Answer: no; it has high accuracy here but zero fraud recall.
2. “Why not pick the model with the highest recall?” Answer: false alarms consume reviewer time; the selected operating point balances assumed costs within a daily review budget.

**Fallback if a cell fails**

Keep an untouched copy of the executed notebooks before presenting. Show its saved result, state that the live rerun failed, and continue the explanation. For a missing variable, restart the notebook and Run All before the talk; during the talk use saved outputs. For a busy port, use the recorded MLflow comparison/alias tables instead of the UI. If the gate rejects an edited model, show the reason; do not weaken thresholds to force a pass. Troubleshoot after the session with `python scripts/run_all.py` and `mlflow-ui.log`.
