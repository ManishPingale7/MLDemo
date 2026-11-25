# 30-minute talk notes

Before presenting: run `python scripts/run_all.py`, open both notebooks and Run All to initialise their kernels. Keep an untouched copy with outputs. Measured notebook runtimes: 10.93s and 55.44s on the tested Linux machine, not a measured Windows laptop.

| Time | Section and what to run | What to say |
|---|---|---|
| 0–2 min | ML 1: saved baseline table/chart | “Approving everything looks accurate because fraud is rare. On validation it is 98.36% accurate and catches zero fraud; precision measures useful alerts, recall measures fraud caught.” |
| 2–3 | ML 2: saved head/types | “This is one synthetic table with 50,150 raw rows, including duplicates. Clear synthetic risk patterns make the workflow easy to demonstrate; all patterns and costs are assumptions.” |
| 3–5 | ML 3: run cleaning summaries live | “We remove duplicate and invalid rows, then standardise categories. The median comes only from training data; future rows cannot influence it.” |
| 5–8 | ML 4: use the five saved figures | “These figures use the training period only. The risky profile combines unusually large amounts, a different city and a new merchant. Several clues together make it easy for the trees to identify.” |
| 8–9 | ML 5: show saved feature list | “Relative amount compares a payment with that customer's usual spend. Each of these four features uses only the current row.” |
| 9–10 | ML 6: show timeline | “We train on the first 60 days, select on the next 15 and test on the last 15. A random split can mix behaviour from the past and future.” |
| 10–12 | ML 7: rerun fit cell live | “The same preprocessing structure supports three models. Only logistic regression needs numeric scaling; all fitted preprocessing comes from training rows.” |
| 12–14 | ML 8: show saved comparison/PR curves | “Random Forest reaches 96.30% precision, 95.59% recall and 0.950 PR-AUC on validation. We compare all models on the same dates.” |
| 14–16 | ML 9: run selection cell live | “We choose Random Forest at 0.60, with precision 96.30%, recall 95.59% and INR 2.23 per transaction. The threshold must stay within 45 alerts on every validation day.” |
| 16–17 | ML 10: use saved final test; do not rerun live | “The frozen choice reaches 97.39% recall, 91.06% precision and 0.938 PR-AUC on later transactions. These are synthetic results, not evidence of production performance.” |
| 17–19 | MLOps 1: run Git demo live | “A branch lets us make a small config change and merge it. A commit identifies exactly which files we used; this demo changes only a note.” |
| 19–22 | MLOps 2: rerun tracking/UI cells live | “An experiment is a saved record of settings, results and a fitted model. These three runs use local SQLite and files; the Git SHA and config hash tell us where they came from.” |
| 22–25 | MLOps 3: run registration, promotion and rollback live | “A registered version is fixed; an alias is a movable name. We move champion to the challenger and back without retraining or deploying.” |
| 25–28 | MLOps 4: run pytest and paired gates live | “Tests check code behaviour; a quality gate checks model acceptability. The real model exits 0, while the baseline exits 1 for low PR-AUC, zero recall and excessive cost.” |
| 28–30 | MLOps 5: show saved workflow/local CI and diagram | “A future GitHub push runs these same checks. Fresh CI compares against a bootstrap baseline; choosing a registry alias is still a separate decision.” |

**Optional 7–10 minute extension:** open the dashboard, start balanced traffic, inspect a request, try a manual transaction, show a blocked baseline release and open GitHub Actions; follow docs/demo_guide.md.

**Two audience questions**

1. “Would you trust a model that approves every transaction?” Answer: no; it has high accuracy here but zero fraud recall.
2. “Why not pick the model with the highest recall?” Answer: false alarms consume reviewer time; the selected operating point balances assumed costs within a daily review budget.

**Fallback if a cell fails**

Show the saved output and state that the live rerun failed. For missing variables or a busy UI port, continue with the saved tables. If a gate blocks an edited model, explain its reason rather than weakening the limits. Troubleshoot afterwards with `python scripts/run_all.py` and `mlflow-ui.log`.
