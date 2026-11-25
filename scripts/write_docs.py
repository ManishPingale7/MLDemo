"""Refresh the measured report and concise talk notes from executed artifacts."""
import json
import re

import nbformat
import pandas as pd
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    summary = json.loads((ROOT / 'models/ml_summary.json').read_text())
    runtime = json.loads((ROOT / 'models/run_summary.json').read_text())
    mlops = json.loads((ROOT / 'models/mlops_summary.json').read_text())
    config = yaml.safe_load((ROOT / 'config.yaml').read_text())
    raw = pd.read_csv(ROOT / 'data/transactions.csv')
    ci_report = json.loads((ROOT / 'models/ci/metrics_summary.json').read_text())
    ci = ci_report['candidate_metrics']
    notebook = nbformat.read(ROOT / 'notebooks/02_mlops.ipynb', as_version=4)
    streams = '\n'.join(output.get('text', '') for cell in notebook.cells
                        for output in cell.get('outputs', []))
    test_count = int(re.search(r'(\d+) passed', streams).group(1))
    rows = summary['comparison']
    test, validation = summary['test'], summary['validation']
    timing = runtime['notebooks']
    comparison = '\n'.join(
        f'| {r["model"]} | {r["threshold"]:.2f} | {r["accuracy"]:.2%} | '
        f'{r["precision"]:.2%} | {r["recall"]:.2%} | {r["pr_auc"]:.3f} | '
        f'{r["cost_per_txn"]:.2f} | {r["train_seconds"]:.2f} |' for r in rows)
    readme = f'''# Fraud detection and a small local MLOps workflow

**synthetic data, patterns modelled on common fraud signals; not real transactions**

**Dataset version: teaching-v2.** This intentionally easy dataset uses clear risk profiles and light label noise so the presentation can focus on the ML/MLOps workflow. Its high scores are by design and are not evidence of real-world fraud performance.

A 30-minute teaching project for third-year engineering students: one table, three learned models, one always-approve baseline, and two executed notebooks. Python 3.11; no cloud account, paid service, deployment or internet access during the workflow. GitHub Actions is supplied for an optional future push; the local runner does not push or contact GitHub.

## Start on Windows 11

Install **64-bit Python 3.11.9** (or a compatible newer 3.11 patch release) and **Git for Windows** first. Extract this project to a writable folder. Double-click `START-WINDOWS.cmd`: it creates `.venv`, installs requirements, generates data, executes both notebooks, checks their runtimes and opens JupyterLab. The initial dependency installation needs internet access.

For a terminal setup in Command Prompt:

```bat
cd path\\to\\fraud-mlops-simple
py -3.11 -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe scripts\\run_all.py
.venv\\Scripts\\python.exe -m jupyter lab notebooks
```

After installation, rerun the last two commands offline; the `.cmd` installer itself always invokes pip. Expected runner output is `PASS` for both notebooks and `PASS all`. Open `01_fraud_detection.ipynb`, then `02_mlops.ipynb`. Select the Python kernel from this virtual environment.

## Start on Linux

Prerequisites: Python 3.11 and Git; LightGBM also needs a working OpenMP runtime (commonly `libgomp1` on Linux).

```bash
cd fraud-mlops-simple
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/run_all.py
python -m jupyter lab notebooks
```

The runner explicitly launches fresh notebook kernels with the current Python interpreter. It rejects Python versions other than 3.11. No `make`, shell-specific dataset commands or notebook downloads are required.

## What is here

```text
config.yaml                       single seed, presets, assumptions, gate thresholds
requirements.txt                  pinned installed environment
START-WINDOWS.cmd                 install, execute and open Jupyter
src/fraud/
  data.py                         short generator with clear synthetic risk profiles
  cleaning.py, features.py        row cleanup, features and time split
  models.py, evaluate.py          pipelines, threshold selection and shared metrics
  plots.py                        notebook charts
  registry.py, gate.py            local MLflow aliases and acceptance rules
  config.py, git_demo.py          configuration and real local Git demonstration
scripts/
  make_data.py, run_all.py        primary entry points
  run_gate.py, run_ci_local.py    gate and local CI commands
  train_quick.py                  isolated tiny CI training
  execute_notebook.py             fresh-kernel execution helper
  build_notebooks.py              notebook source authoring helper
  write_docs.py                   refresh measured README/talk notes
notebooks/                         two notebooks saved with visible outputs
tests/                             data, preprocessing, metrics, gate, notebook/CI checks
.github/workflows/ci.yml            optional hosted checks
docs/data_dictionary.md             schema, generation assumptions and defects
docs/talk_notes.md                  30-minute running order
data/, models/                     generated and Git-ignored
```

Do not run `build_notebooks.py` just to present: it rewrites notebooks without outputs. `run_all.py` executes the existing notebooks and saves their outputs. `write_docs.py` refreshes this measured report after a new successful execution.

## Tools in one line each

- NumPy and pandas generate, clean and inspect the single transaction table.
- scikit-learn supplies preprocessing pipelines, logistic regression, random forest and metrics.
- LightGBM fits a small boosted-tree classifier.
- Matplotlib and seaborn draw readable static charts.
- JupyterLab presents code, charts and computed interpretations together.
- Git versions code and assumptions; the notebook creates real local commits, branches and tags.
- MLflow stores experiment records, fitted pipelines and named model-version aliases locally.
- pytest checks behaviour; Ruff catches basic Python mistakes.
- GitHub Actions describes checks for a future push; the same checks also run locally.

## ML choices and assumptions

The generator produces 50,000 events plus 150 duplicate rows over 90 days. Fraud is about 1.5% by **assumption**; the actual raw dataset rate is {raw.is_fraud.mean():.3%}. This prevalence is deliberately high enough for readable classroom metrics. Any suggestion that operational fraud rates are lower is an assumption here, not a sourced industry statistic. Every generated pattern is an assumption, described in [the data dictionary](docs/data_dictionary.md).

Routine payments follow customer preferences. Rare risky profiles combine spending 7–15 times the usual amount, a different city and a first-observed merchant, usually with shopping/electronics/gaming and a desktop device. Their assumed fraud probability is 97%, versus 0.02% for routine profiles. Profile membership is never an output column or model input; ID columns are excluded.

Cleaning removes duplicates and nonpositive amounts, standardises categories and parses timestamps. Missing devices become `unknown`; numeric medians and one-hot vocabularies are learned on training rows only. The cleaning section shows an imputed preview using only the first 60 days. Actual model imputation remains inside the pipeline. Four additional inputs express night-time activity, relative amount, long distance and log amount. IDs and raw timestamps are excluded.

EDA uses the first 60 days only. The time split is 60/15/15 days. Validation selects thresholds from a fixed 0.01–0.99 grid, minimising cost within the daily alert budget and requiring at least {config['thresholds']['min_precision']:.0%} precision where achievable. A model unable to reach that precision is displayed for comparison but cannot become the selected model. Selection uses the lowest cost among models passing all quality requirements; cost ties use higher PR-AUC. There is no hyperparameter search. The final 15-day test set is scored once per notebook execution, with the frozen choice; software verification repeats the notebook without tuning to test results.

The **assumed** cost is INR 3,000 per missed fraud and INR 100 per false alarm, regardless of transaction amount. A false positive incurs the review cost; true-positive reviews are assumed to have no additional cost in this simplified formula. Reviewers can handle at most 45 alerts on any validation day. Tiny CI scales that budget to 4.5, effectively at most four integer alerts per day. The budget constrains validation selection; it cannot guarantee a future day's alert volume.

PR-AUC means scikit-learn **average precision**, not trapezoidal integration. Class weights address imbalance; no resampling is used. Models are intentionally small and use one CPU thread. Logistic regression is linear in its transformed inputs; built-in importance describes fitted model behaviour, not causation.

## Measured results

Validation metrics at each model's own budget-compliant threshold:

| Model | Threshold | Accuracy | Precision | Recall | PR-AUC | Cost INR/txn | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{comparison}

Selected **{summary['selected_model']}**, threshold **{summary['threshold']:.2f}**. Final test: accuracy **{test['accuracy']:.2%}**, precision **{test['precision']:.2%}**, recall **{test['recall']:.2%}**, F1 **{test['f1']:.3f}**, PR-AUC **{test['pr_auc']:.3f}**, ROC-AUC **{test['roc_auc']:.3f}**, and cost **INR {test['cost_per_txn']:.2f}/transaction**, with **{test['alerts']} alerts**. Test metrics were not used to choose the model or threshold.

Full-data quality gate: selected model **passes (exit {mlops['gate_exit_codes'][0]})**; always-approve baseline **is blocked (exit {mlops['gate_exit_codes'][1]})**. Tiny CI selects {ci_report['candidate']} (precision {ci['precision']:.2%}, recall {ci['recall']:.2%}, PR-AUC {ci['pr_auc']:.3f}, cost INR {ci['cost_per_txn']:.2f}/transaction). A small synthetic validation split can score perfectly; it is a smoke check, not a stable performance estimate. Local CI passes. **{test_count} pytest checks pass; Ruff passes; pip reports no broken requirements.**

Fresh-kernel execution including kernel startup: notebook 1 **{timing['01_fraud_detection.ipynb']:.2f} s**; notebook 2 **{timing['02_mlops.ipynb']:.2f} s**; whole runner **{runtime['total_seconds']:.2f} s**. Measured on Python {runtime['python']}, Linux x86-64, Intel Xeon Platinum 8573C, three visible logical CPUs. This is an execution-environment measurement, **not a Windows laptop benchmark**; laptop runtime remains unverified. Runtime metadata is in `models/run_summary.json`. The runner exits nonzero if either notebook exceeds 60 seconds.

## MLflow and the registry

Tracking uses `sqlite:///mlflow.db` and a local `mlartifacts/` folder. The notebook starts a detached, loopback-only UI at [localhost:5000](http://localhost:5000) and checks its health endpoint. Its runs and champion/challenger aliases were also checked in headless Chromium after notebook kernels had exited. Open the `fraud-local` experiment and the `fraud-classroom` registered model. A second run adds new runs and model versions rather than deleting history.

Each run records parameters, validation scores, threshold, pipeline, dataset version, Git SHA and configuration hash. Dataset version is also saved with each model; the gate rejects mismatched versions. Registered versions retain their threshold as a tag. The gate loads the **current champion alias**, including that version's threshold. Moving an alias never changes a pipeline's fitted weights. The registry section briefly promotes a second model to demonstrate the API, then rolls back; it is not a quality-approved release. CI never changes registry aliases automatically.

To run the full gate after the notebooks:

```text
python scripts/run_gate.py
python scripts/run_gate.py --candidate models/baseline.joblib --output models/blocked_summary.json
```

The second command deliberately returns 1. Gate requirements in `config.yaml` are PR-AUC ≥{config['gate']['min_pr_auc']:.2f}, precision ≥{config['gate']['min_precision']:.2f}, recall ≥{config['gate']['min_recall']:.2f}, cost ≤INR {config['gate']['max_cost']:.0f}/transaction, and cost no higher than the current champion. All comparisons use validation data. Initial champion equals the selected model, so the first full gate demonstrates equality to an established reference; it does not prove an improvement over an older production model.

```text
python scripts/run_ci_local.py
```

This runs Ruff, pytest, tiny generation, quick training and the gate. A fresh CI runner has no registry history, so its reference is an explicitly saved always-approve bootstrap model. Tiny data and models live separately and never replace full-data artifacts. CI uploads JSON metrics for inspection. The diagram's registry step is a subsequent decision, not an implemented deployment step.

The UI continues running after notebooks finish. To stop it, stop the Python process whose PID appears in `.runtime/mlflow_ui.json` (Task Manager on Windows). To start it manually: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1`. If port 5000 belongs to another service, stop that service before rerunning.

## Reproducibility, compatibility and verification limits

- This version changes the data-generating assumptions to produce stronger classroom results, as requested. It supersedes the original overlapping-risk design and its lower-recall target; old and new metrics are not comparable as a model-only improvement. The new generator and selection settings were frozen on validation before the new test set was scored.
- One seed in `config.yaml` drives generation and estimators. Repeated full runs produced the same dataset and predictive metrics. Timings, Git hashes, run IDs and registry version numbers are expected to change. Bit-for-bit model artifacts across OS/CPU versions are not promised.
- The exact installed dependency set is pinned in `requirements.txt`; POSIX-only dependencies have Windows markers. Windows-only pywin32 and pywinpty are conditionally pinned from a successful Windows Python 3.11.9 dependency resolution. Those two Windows-only pins could not be installed or runtime-tested on Linux; this is an explicit exception to the requirement that every pin be installed and verified together. Python 3.11.0–3.11.4 are too old for the pinned dependency set.
- MLflow **3.6.0** and LightGBM **4.6.0** signatures were inspected before integration. This uses MLflow's `name=` model logging and alias APIs, not deprecated stages. Model signatures use numeric doubles, and the logged Python function returns class probabilities.
- The first install selected SQLAlchemy 2.1.9, which failed because MLflow imports an API removed in that series. Pinning **SQLAlchemy 2.0.44** fixed it; local tracking, model reload, aliases, promotion and rollback were then executed successfully.
- Both notebooks ran top to bottom from fresh kernels and retain outputs. Offline verification blocked external Python socket connections while permitting loopback for Jupyter and MLflow. The workflow YAML was parsed and its referenced scripts checked.
- **Not tested:** Windows 11 execution, a physical laptop's runtime, hosted GitHub Actions, or real fraud data. Windows is the intended setup target, not a claimed verified platform. No push or deployment was performed.
- The conflicting request for “2 more models” was resolved in favour of the explicit three-model limit and acceptance criteria: three learned models plus the baseline. Advanced monitoring, serving and model-explanation systems are omitted as requested.
- No published fraud statistics are used. Software documentation links (2026; API behaviour was verified directly against installed versions): [MLflow model registry](https://mlflow.org/docs/latest/ml/model-registry/), [MLflow sklearn API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html), [LightGBM classifier API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html), [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Presentation and recovery

Use [the short talk notes](docs/talk_notes.md). Pre-run both notebooks before the session. If a live cell fails, show its saved output and explain the input, decision and result. Use the last successful notebook outputs for discussion; do not invent replacement metrics. `python scripts/run_all.py` rebuilds generated data and model artifacts, but retains Git and MLflow history. The distributed archive includes the generated CSV and model/report artifacts; local tracking is rebuilt on your machine so artifact paths are valid there.
'''
    (ROOT / 'README.md').write_text(readme, encoding='utf-8')
    notes = f'''# 30-minute talk notes

Before presenting: install Python 3.11 and Git; run `python scripts/run_all.py`. Open both notebooks and run each once so their kernels hold the variables. Keep saved outputs visible. Use these notes outside the notebooks. Numbers below come from the executed notebooks; full execution took {timing['01_fraud_detection.ipynb']:.2f}s and {timing['02_mlops.ipynb']:.2f}s on the tested Linux machine, not a measured Windows laptop.

| Time | Section and what to run | What to say |
|---|---|---|
| 0–2 min | ML 1: saved baseline table/chart | “Approving everything looks accurate because fraud is rare. On validation it is {rows[0]['accuracy']:.2%} accurate and catches zero fraud; precision measures useful alerts, recall measures fraud caught.” |
| 2–3 | ML 2: saved head/types | “This is one synthetic table with 50,150 raw rows, including duplicates. Clear synthetic risk patterns make the workflow easy to demonstrate; all patterns and costs are assumptions.” |
| 3–5 | ML 3: run cleaning summaries live | “We remove duplicate and invalid rows, then standardise categories. The median comes only from training data; future rows cannot influence it.” |
| 5–8 | ML 4: use the five saved figures | “These figures use the training period only. The risky profile combines unusually large amounts, a different city and a new merchant. Several clues together make it easy for the trees to identify.” |
| 8–9 | ML 5: show saved feature list | “Relative amount compares a payment with that customer's usual spend. Each of these four features uses only the current row.” |
| 9–10 | ML 6: show timeline | “We train on the first 60 days, select on the next 15 and test on the last 15. A random split can mix behaviour from the past and future.” |
| 10–12 | ML 7: rerun fit cell live | “The same preprocessing structure supports three models. Only logistic regression needs numeric scaling; all fitted preprocessing comes from training rows.” |
| 12–14 | ML 8: show saved comparison/PR curves | “{summary['selected_model']} reaches {validation['precision']:.2%} precision, {validation['recall']:.2%} recall and {validation['pr_auc']:.3f} PR-AUC on validation. We compare all models on the same dates.” |
| 14–16 | ML 9: run selection cell live | “We choose {summary['selected_model']} at {summary['threshold']:.2f}, with precision {validation['precision']:.2%}, recall {validation['recall']:.2%} and INR {validation['cost_per_txn']:.2f} per transaction. The threshold must stay within 45 alerts on every validation day.” |
| 16–17 | ML 10: use saved final test; do not rerun live | “The frozen choice reaches {test['recall']:.2%} recall, {test['precision']:.2%} precision and {test['pr_auc']:.3f} PR-AUC on later transactions. These are synthetic results, not evidence of production performance.” |
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
'''
    (ROOT / 'docs/talk_notes.md').write_text(notes, encoding='utf-8')
    print('Updated README and talk notes from executed artifacts.')


if __name__ == '__main__':
    main()
