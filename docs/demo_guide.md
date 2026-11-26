# Give the complete demo

This project now covers model training, experiment tracking, quality checks, GitHub CI, a manual local release, FastAPI serving and a browser dashboard. The model and traffic are synthetic teaching examples. There is no public prediction service or automatic production deployment.

## 1. Prepare your Windows laptop once

Install 64-bit Python 3.11.9 (or a compatible newer 3.11 patch) and Git for Windows. Extract the project ZIP into a normal writable folder, for example `C:\fraud-demo\fraud-mlops-simple`.

Double-click `START-WINDOWS.cmd` to install packages, execute the two notebooks and open JupyterLab. Internet is needed for this first installation. Before presenting, open both notebooks and use **Run → Run All Cells**, first notebook 1 and then notebook 2, to initialise their kernels.

Double-click `START-DEMO.cmd` to install/check packages and open the dashboard. It starts a quality-gated local model release if none exists. Keep its terminal open. The dashboard is at **http://127.0.0.1:8765**; the offline API reference is at **http://127.0.0.1:8765/docs**.

For later offline sessions, skip pip installation:

```bat
cd /d C:\fraud-demo\fraud-mlops-simple
.venv\Scripts\python.exe -m jupyter lab notebooks
```

In a second Command Prompt:

```bat
cd /d C:\fraud-demo\fraud-mlops-simple
.venv\Scripts\python.exe scripts\run_demo.py
```

If you cloned the GitHub repository instead of downloading the prepared ZIP, generated data and models are absent. The demo launcher runs the notebooks to create them on first use. Git must be installed for that preparation step.

## 2. Show the two notebooks

Use notebook 1 for the dataset, cleaning, time split, comparison and selected model. Use notebook 2 for Git, MLflow, registry aliases, tests, the gate and CI. The short talk notes cover their original 30-minute sequence. Allow another 7–10 minutes for the dashboard and serving demonstration below.

MLflow's UI on port 5000 tracks experiments and model versions. The dashboard on port 8765 sends requests to the prediction API. They are separate applications.

## 3. Show predictions happening

1. Open the dashboard. The top-right badge should say **API online**.
2. Keep **Balanced demo · 50% fraud** selected, then click **Start traffic**. This mode deliberately shows both outcomes quickly.
3. Explain the feed: a transaction goes to `POST /predict`; the fitted pipeline returns a score; a score at or above the stored threshold is flagged for review.
4. Click **Pause traffic**. Click a row to inspect the actual JSON request and response. Point out that `is_fraud` is not in the request.
5. Select **Try a transaction → Everyday purchase → Get prediction**. Then repeat with **Unusual purchase**. The exact response is calculated by the active model, not hardcoded in the UI.
6. Return to **Natural mix · dataset rate** and click **Start traffic**. Most payments are now legitimate, as in the generated validation period. Changing the mix resets session counters.
7. Click **Export session CSV** to save the requests' scores and decisions. The replay stops at 1,000 requests; reset for another session.

Session precision and recall use only replay transactions with known synthetic labels. Manual inputs have no known outcome, so they are excluded. Balanced replay has an artificial 50% fraud rate: its session precision is not the model's held-out test precision. A session with no labelled alerts or no known fraud displays a dash instead of an invented score.

Traffic samples come only from validation data, never the final test period. Replaying the same mode from offset zero returns the same sequence. Request latency and clock time are naturally variable. Speed controls are targets; requests are sequential and slow down if the model is busy.

## 4. Show release controls

In an activated project terminal:

```text
python scripts/deploy_model.py
```

This evaluates the candidate on validation data and compares it with the current local release. The first release compares with the always-approve baseline. It checks precision, recall, PR-AUC, cost and dataset version, copies the fitted model into `models/deployed/`, and atomically updates `active.json`.

Click **Reload model** in the dashboard. The release number changes; the session resets so metrics do not mix model releases. Requests already in progress finish on their original model snapshot.

Now demonstrate a rejected release:

```text
python scripts/deploy_model.py --candidate models/baseline.joblib
```

Expected: `BLOCK` and exit code 1. The currently active release stays unchanged.

If you have created at least two releases, demonstrate rollback:

```text
python scripts/deploy_model.py --rollback
```

Click **Reload model** again. Releasing the same fitted model twice is fine for demonstrating version mechanics; it does not represent a new learned model.

MLflow's `champion` alias and the API's local release pointer are distinct. The notebook shows registry operations; `deploy_model.py` is the explicit, gated step that changes what the API can load. An alias change alone never deploys a model.

## 5. Show actual GitHub CI

Repository: https://github.com/ManishPingale7/MLDemo

The [first complete hosted run](https://github.com/ManishPingale7/MLDemo/actions/runs/45666369894) passed on Linux and Windows. Open **Actions → Fraud model and API checks**. A push or pull request triggers separate Linux and Windows jobs. Each job installs the pinned dependencies, runs Ruff and pytest, generates tiny data, trains a candidate, runs the quality gate, packages a local release and exercises the API.

Download the **model-and-metrics** artifact from a successful job to see its tiny model release and JSON reports. This artifact is a CI smoke-test model, not a replacement for the full-data presentation model. GitHub does not automatically deploy anything to your laptop.

For an ordinary future code change in your clone:

```text
git add .
git commit -m "Update fraud demo"
git push origin main
```

Use a feature branch and pull request for collaborative changes. Re-run the same checks locally with `python scripts/run_ci_local.py` before pushing.

## 6. Optional Docker and DVC extension

Follow [the Docker/DVC guide](docker_dvc_demo.md): run the same model in a container,
then version and restore a dataset copy. Allow seven extra minutes and pre-build the
image before presenting. The two notebooks remain the core 30-minute presentation.

## Troubleshooting and boundaries

- No model/data: run `python scripts/run_all.py`, then `python scripts/run_demo.py`.
- Port 8765 occupied: use `python scripts/run_demo.py --port 8766` and the printed URL.
- Gate blocks a model: read the failed requirement. Do not lower limits to force the demonstration.
- Model files changed/corrupt: create a fresh release; checksum or configuration mismatches are refused.
- Browser loses API connection: check the server terminal, then refresh. Ctrl+C stops the service.
- Model scores are not calibrated fraud probabilities. Latency is an observed request measurement, not a production benchmark.
- The default server binds to loopback. Authentication, rate limiting, a public host, drift monitoring and automated production CD are not implemented; they are beyond this laptop demo.
- Visual browser QA was unavailable in the build environment because its managed preview relay failed. HTTP/API tests and JavaScript syntax checks are separate verification; rehearse the UI on your own browser before presenting.
