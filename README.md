# Fraud detection and a small local MLOps workflow

**synthetic data, patterns modelled on common fraud signals; not real transactions**

**Dataset version: teaching-v2.** This intentionally easy dataset uses clear risk profiles and light label noise so the presentation can focus on the ML/MLOps workflow. Its high scores are by design and are not evidence of real-world fraud performance.

A 30-minute teaching project for third-year engineering students: one table, three learned models, one always-approve baseline, and two executed notebooks. Python 3.11; model training, tracking and the live dashboard run locally with no paid service. GitHub Actions runs hosted CI after a push. Serving uses a manual, quality-gated local release; there is no automatic production deployment.

## Live dashboard and FastAPI

Double-click **START-DEMO.cmd** after extracting the project, or run `python scripts/run_demo.py` in the installed environment. Open **http://127.0.0.1:8765**. The dashboard provides traffic replay, start/pause and speed controls, a score chart, a decision feed, request inspection, custom inputs, CSV export and explicit model reload. The [step-by-step demo guide](docs/demo_guide.md) explains setup and the presentation.

FastAPI serves the same fitted pipeline as the notebooks through `POST /predict`. Inputs are validated; labels are rejected; missing values use the training-fitted imputer. The default API documentation normally loads online assets, so this project supplies an offline reference at `/docs` and the schema at `/openapi.json`.

`python scripts/deploy_model.py` gates a candidate against the current local release before updating `models/deployed/active.json`; `--rollback` restores the preceding release. The running API changes only after **Reload model**. A checksum validates each artifact. This is manual local deployment, distinct from the MLflow registry alias demonstration.

Traffic replays labelled **validation** transactions. Balanced mode deliberately uses 50% fraud to make both decisions visible; natural mode replays the validation distribution. Session metrics use only labelled replay requests, excluding manual inputs. Neither mode changes the original held-out test results. No frontend build, Node installation, CDN, cloud prediction endpoint or external font is needed on the laptop.

Repository and hosted CI: [ManishPingale7/MLDemo](https://github.com/ManishPingale7/MLDemo) · [Actions](https://github.com/ManishPingale7/MLDemo/actions). The workflow tests Linux and Windows, trains a tiny model, runs the gate, creates a release and exercises the API. Uploaded models are CI smoke-test artifacts; they are not automatically served to users.

## Start on Windows 11

Install **64-bit Python 3.11.9** (or a compatible newer 3.11 patch release) and **Git for Windows** first. Extract this project to a writable folder. Double-click `START-WINDOWS.cmd`: it creates `.venv`, installs requirements, generates data, executes both notebooks, checks their runtimes and opens JupyterLab. The initial dependency installation needs internet access.

For a terminal setup in Command Prompt:

```bat
cd path\to\fraud-mlops-simple
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe scripts\run_all.py
.venv\Scripts\python.exe -m jupyter lab notebooks
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
START-DEMO.cmd                    install and open the live dashboard
web/                              offline HTML, CSS and JavaScript interface
src/fraud/
  data.py                         short generator with clear synthetic risk profiles
  cleaning.py, features.py        row cleanup, features and time split
  models.py, evaluate.py          pipelines, threshold selection and shared metrics
  plots.py                        notebook charts
  registry.py, gate.py            local MLflow aliases and acceptance rules
  config.py, git_demo.py          configuration and real local Git demonstration
  api.py, serving.py              validated FastAPI requests and fitted-model inference
  deployment.py                   quality-gated local releases and rollback
scripts/
  make_data.py, run_all.py        primary entry points
  run_gate.py, run_ci_local.py    gate and local CI commands
  train_quick.py                  isolated tiny CI training
  execute_notebook.py             fresh-kernel execution helper
  build_notebooks.py              notebook source authoring helper
  write_docs.py                   refresh measured README/talk notes
  run_demo.py, deploy_model.py     dashboard launcher and local releases
  smoke_api.py                    exercise a deployed model through API requests
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
- FastAPI and Uvicorn serve predictions and the dashboard on localhost.
- GitHub Actions runs the checks after a push; the same checks also run locally.

## ML choices and assumptions

The generator produces 50,000 events plus 150 duplicate rows over 90 days. Fraud is about 1.5% by **assumption**; the actual raw dataset rate is 1.490%. This prevalence is deliberately high enough for readable classroom metrics. Any suggestion that operational fraud rates are lower is an assumption here, not a sourced industry statistic. Every generated pattern is an assumption, described in [the data dictionary](docs/data_dictionary.md).

Routine payments follow customer preferences. Rare risky profiles combine spending 7–15 times the usual amount, a different city and a first-observed merchant, usually with shopping/electronics/gaming and a desktop device. Their assumed fraud probability is 97%, versus 0.02% for routine profiles. Profile membership is never an output column or model input; ID columns are excluded.

Cleaning removes duplicates and nonpositive amounts, standardises categories and parses timestamps. Missing devices become `unknown`; numeric medians and one-hot vocabularies are learned on training rows only. The cleaning section shows an imputed preview using only the first 60 days. Actual model imputation remains inside the pipeline. Four additional inputs express night-time activity, relative amount, long distance and log amount. IDs and raw timestamps are excluded.

EDA uses the first 60 days only. The time split is 60/15/15 days. Validation selects thresholds from a fixed 0.01–0.99 grid, minimising cost within the daily alert budget and requiring at least 85% precision where achievable. A model unable to reach that precision is displayed for comparison but cannot become the selected model. Selection uses the lowest cost among models passing all quality requirements; cost ties use higher PR-AUC. There is no hyperparameter search. The final 15-day test set is scored once per notebook execution, with the frozen choice; software verification repeats the notebook without tuning to test results.

The **assumed** cost is INR 3,000 per missed fraud and INR 100 per false alarm, regardless of transaction amount. A false positive incurs the review cost; true-positive reviews are assumed to have no additional cost in this simplified formula. Reviewers can handle at most 45 alerts on any validation day. Tiny CI scales that budget to 4.5, effectively at most four integer alerts per day. The budget constrains validation selection; it cannot guarantee a future day's alert volume.

PR-AUC means scikit-learn **average precision**, not trapezoidal integration. Class weights address imbalance; no resampling is used. Models are intentionally small and use one CPU thread. Logistic regression is linear in its transformed inputs; built-in importance describes fitted model behaviour, not causation.

## Measured results

Validation metrics at each model's own budget-compliant threshold:

| Model | Threshold | Accuracy | Precision | Recall | PR-AUC | Cost INR/txn | Fit seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Always approve | 0.01 | 98.36% | 0.00% | 0.00% | 0.016 | 49.26 | 0.10 |
| Logistic Regression | 0.85 | 99.61% | 86.11% | 91.18% | 0.935 | 4.59 | 0.25 |
| Random Forest | 0.60 | 99.87% | 96.30% | 95.59% | 0.950 | 2.23 | 1.34 |
| LightGBM | 0.79 | 99.87% | 96.30% | 95.59% | 0.918 | 2.23 | 0.38 |

Selected **Random Forest**, threshold **0.60**. Final test: accuracy **99.83%**, precision **91.06%**, recall **97.39%**, F1 **0.941**, PR-AUC **0.938**, ROC-AUC **0.985**, and cost **INR 1.24/transaction**, with **123 alerts**. Test metrics were not used to choose the model or threshold.

Full-data quality gate: selected model **passes (exit 0)**; always-approve baseline **is blocked (exit 1)**. Tiny CI selects Random Forest (precision 100.00%, recall 100.00%, PR-AUC 1.000, cost INR 0.00/transaction). A small synthetic validation split can score perfectly; it is a smoke check, not a stable performance estimate. Local CI passes. **40 pytest checks pass; Ruff passes; pip reports no broken requirements.**

Fresh-kernel execution including kernel startup: notebook 1 **10.93 s**; notebook 2 **55.44 s**; whole runner **67.62 s**. Measured on Python 3.11.18, Linux x86-64, Intel Xeon Platinum 8573C, three visible logical CPUs. This is an execution-environment measurement, **not a Windows laptop benchmark**; laptop runtime remains unverified. Runtime metadata is in `models/run_summary.json`. The runner exits nonzero if either notebook exceeds 60 seconds.

## MLflow and the registry

Tracking uses `sqlite:///mlflow.db` and a local `mlartifacts/` folder. The notebook starts a detached, loopback-only UI at [localhost:5000](http://localhost:5000) and checks its health endpoint. Its runs and champion/challenger aliases were also checked in headless Chromium after notebook kernels had exited. Open the `fraud-local` experiment and the `fraud-classroom` registered model. A second run adds new runs and model versions rather than deleting history.

Each run records parameters, validation scores, threshold, pipeline, dataset version, Git SHA and configuration hash. Dataset version is also saved with each model; the gate rejects mismatched versions. Registered versions retain their threshold as a tag. The gate loads the **current champion alias**, including that version's threshold. Moving an alias never changes a pipeline's fitted weights. The registry section briefly promotes a second model to demonstrate the API, then rolls back; it is not a quality-approved release. CI never changes registry aliases automatically.

To run the full gate after the notebooks:

```text
python scripts/run_gate.py
python scripts/run_gate.py --candidate models/baseline.joblib --output models/blocked_summary.json
```

The second command deliberately returns 1. Gate requirements in `config.yaml` are PR-AUC ≥0.85, precision ≥0.85, recall ≥0.85, cost ≤INR 10/transaction, and cost no higher than the current champion. All comparisons use validation data. Initial champion equals the selected model, so the first full gate demonstrates equality to an established reference; it does not prove an improvement over an older production model.

```text
python scripts/run_ci_local.py
```

This runs Ruff, pytest, tiny generation, quick training and the gate. A fresh CI runner has no registry history, so its reference is an explicitly saved always-approve bootstrap model. Tiny data and models live separately and never replace full-data artifacts. CI uploads JSON metrics and its tiny released model for inspection, and runs an API smoke check. The diagram's registry step is a subsequent decision, not an implemented deployment step.

The UI continues running after notebooks finish. To stop it, stop the Python process whose PID appears in `.runtime/mlflow_ui.json` (Task Manager on Windows). To start it manually: `python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000 --workers 1`. If port 5000 belongs to another service, stop that service before rerunning.

## Reproducibility, compatibility and verification limits

- This version changes the data-generating assumptions to produce stronger classroom results, as requested. It supersedes the original overlapping-risk design and its lower-recall target; old and new metrics are not comparable as a model-only improvement. The new generator and selection settings were frozen on validation before the new test set was scored.
- One seed in `config.yaml` drives generation and estimators. Repeated full runs produced the same dataset and predictive metrics. Timings, Git hashes, run IDs and registry version numbers are expected to change. Bit-for-bit model artifacts across OS/CPU versions are not promised.
- The exact installed dependency set is pinned in `requirements.txt`; POSIX-only dependencies have Windows markers. Windows-only pywin32 and pywinpty are conditionally pinned from a successful Windows Python 3.11.9 dependency resolution. Those two Windows-only pins could not be installed or runtime-tested on Linux; this is an explicit exception to the requirement that every pin be installed and verified together. Python 3.11.0–3.11.4 are too old for the pinned dependency set.
- MLflow **3.6.0** and LightGBM **4.6.0** signatures were inspected before integration. This uses MLflow's `name=` model logging and alias APIs, not deprecated stages. Model signatures use numeric doubles, and the logged Python function returns class probabilities.
- The first install selected SQLAlchemy 2.1.9, which failed because MLflow imports an API removed in that series. Pinning **SQLAlchemy 2.0.44** fixed it; local tracking, model reload, aliases, promotion and rollback were then executed successfully.
- Both notebooks ran top to bottom from fresh kernels and retain outputs. Offline verification blocked external Python socket connections while permitting loopback for Jupyter and MLflow. The workflow YAML was parsed and its referenced scripts checked.
- **Not tested:** Windows 11 execution, a physical laptop's runtime, hosted GitHub Actions, or real fraud data. Windows is the intended setup target, not a claimed verified platform. A public repository is used for GitHub CI; prediction serving remains local and manual.
- The conflicting request for “2 more models” was resolved in favour of the explicit three-model limit and acceptance criteria: three learned models plus the baseline. The later request adds a local dashboard, FastAPI serving, manual deployment and hosted CI. Advanced monitoring and model-explanation systems remain omitted.
- No published fraud statistics are used. Software documentation links (2026; API behaviour was verified directly against installed versions): [MLflow model registry](https://mlflow.org/docs/latest/ml/model-registry/), [MLflow sklearn API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.sklearn.html), [LightGBM classifier API](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.LGBMClassifier.html), [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

## Presentation and recovery

Use [the short talk notes](docs/talk_notes.md) for the two notebooks, then [the demo guide](docs/demo_guide.md) for a 7–10 minute dashboard and API extension. Pre-run both notebooks before the session. If a live cell fails, show its saved output and explain the input, decision and result. Use the last successful notebook outputs for discussion; do not invent replacement metrics. `python scripts/run_all.py` rebuilds generated data and model artifacts, but retains Git and MLflow history. The distributed archive includes the generated CSV and model/report artifacts; local tracking is rebuilt on your machine so artifact paths are valid there.
