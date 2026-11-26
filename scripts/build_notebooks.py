"""Author the small notebook cells; executed outputs are saved by run_all.py."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def write(name, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {'display_name': 'Python 3.11 (project)',
                                   'language': 'python', 'name': 'python3'}
    nbf.write(notebook, ROOT / 'notebooks' / name)


def md(text):
    return nbf.v4.new_markdown_cell(text)


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def notebook_one():
    cells = [md('# 1. Problem and metrics\nFlag suspicious payments for review in an intentionally easy synthetic dataset with clear risk profiles and light label noise.'), code('''
from pathlib import Path
import os
import sys
import time
import json
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
started = time.perf_counter()
'''), code('''
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from fraud.config import load_config
from fraud import plots
config = load_config()
plots.setup()
'''), md('Precision is the fraction of alerts that are fraud. Recall is the fraction of fraud caught. PR-AUC here means average precision, a summary of the precision–recall curve across thresholds. Cost is missed fraud times its assumed cost plus false alarms times their assumed cost, divided by transaction count.\n\nAccuracy counts all correct predictions, so the large legitimate class can hide missed fraud.'), code('''
from fraud.evaluate import metrics
raw = pd.read_csv('data/transactions.csv')
from fraud.cleaning import clean_data, time_split, split_summary
clean = clean_data(raw)
past = time_split(clean, config)[0]
approve = pd.DataFrame([metrics(past.is_fraud, np.zeros(len(past)), 0.5, config)], index=['Always approve'])
display(approve[['accuracy', 'precision', 'recall', 'pr_auc', 'cost_per_txn']].round(4))
plots.baseline_chart(approve)
display(Markdown(f'Always approving gets {approve.iloc[0].accuracy:.2%} accuracy but catches {approve.iloc[0].recall:.0%} of fraud in the training period. Assumed costs are INR {config["costs"]["missed_fraud"]:,} per missed fraud and INR {config["costs"]["false_alarm"]:,} per false alarm.'))
'''), md('# 2. Load data\nInspect the one CSV table so its dimensions, types and memory use are explicit.'), code('''
display(raw.head())
display(raw.dtypes.rename('dtype').to_frame())
display(Markdown(f'The raw table has {raw.shape[0]:,} rows and {raw.shape[1]} columns, using {raw.memory_usage(deep=True).sum() / 1024**2:.2f} MiB. Timestamps are strings until cleaning parses them.'))
'''), md('# 3. Clean data\nRemove duplicates and invalid amounts, standardise categories and use training-only imputation to avoid learning from future rows.'), code('''
missing = pd.DataFrame({'missing_count': raw.isna().sum(), 'missing_percent': raw.isna().mean() * 100})
display(missing)
plots.missing_chart(raw)
summary = pd.DataFrame({'before': [len(raw), raw.duplicated().sum(), (raw.amount_inr <= 0).sum(), raw.device_type.isna().sum()], 'after': [len(clean), clean.duplicated().sum(), (clean.amount_inr <= 0).sum(), clean.device_type.isna().sum()]}, index=['rows', 'duplicates', 'invalid amounts', 'missing device'])
display(summary)
display(Markdown(f'Cleaning removes {len(raw)-len(clean):,} rows; {clean.distance_from_home_km.isna().sum():,} distances remain missing until the pipeline learns a median from training rows. Missing devices become “unknown”; category spaces/case are normalised.'))
'''), code('''
training_median = past.distance_from_home_km.median()
imputed_preview = clean[['distance_from_home_km', 'device_type']].copy()
imputed_preview['distance_from_home_km'] = imputed_preview.distance_from_home_km.fillna(training_median)
display(imputed_preview.head())
display(Markdown(f'This preview uses the training-period median of {training_median:.2f} km and has {imputed_preview.isna().sum().sum():,} missing cells. The actual model repeats imputation inside its fitted pipeline; hour and day_of_week come directly from timestamps.'))
'''), md('# 4. EDA\nUse only the first 60 days for these five exploratory figures so the final test period remains untouched.')]
    for number, intro in enumerate(['Count labels to show imbalance.', 'Compare hours to motivate is_night.', 'Compare amounts and their tail to motivate amount_to_avg_ratio and log_amount.', 'Compare categories and devices to inspect categorical risk.', 'Compare distance and merchant familiarity to motivate is_far_from_home.'], 1):
        cells += [md(intro), code(f'display(Markdown(plots.eda_chart(past, {number}, config)))')]
    cells += [md('# 5. Feature engineering\nAdd four row-local features to express the time, relative amount and distance patterns seen in EDA.'), code('''
from fraud.features import add_features, BASE_NUMERIC, ENGINEERED, CATEGORICAL, FEATURES
featured = add_features(clean, config)
display(pd.DataFrame({'feature': ENGINEERED, 'reason': ['Hour chart: before 06:00', 'Amount chart: spend relative to customer usual amount', 'Distance chart: more than configured 200 km', 'Amount chart: compress the long right tail']}))
print('Before:', BASE_NUMERIC + CATEGORICAL)
print('After: ', FEATURES)
display(featured[ENGINEERED].head())
display(Markdown(f'The feature set grows from {len(BASE_NUMERIC+CATEGORICAL)} to {len(FEATURES)} inputs; IDs and the raw timestamp are excluded. No feature uses the fraud label or future transactions.'))
'''), md('# 6. Train/validation/test split\nwe train on the past and test on the future, like real life\n\nA random split can be too optimistic because it mixes earlier and later behaviour; this synthetic dataset does not establish how large that effect would be in production.'), code('''
train, validation, test = time_split(featured, config)
balance = split_summary((train, validation, test))
display(balance)
plots.timeline(balance)
display(Markdown(f'The split contains {len(train):,} training, {len(validation):,} validation and {len(test):,} test rows across 60/15/15 calendar days. The train fraud rate is {train.is_fraud.mean():.2%} versus {validation.is_fraud.mean():.2%} in validation.'))
'''), md('# 7. Preprocessing pipeline and models\nFit one preprocessing-and-model pipeline per estimator on training rows only, using class weights to address imbalance.'), code('''
from fraud.models import fit_models, select_model, train_validation_scores, importance, save_bundle
bundles, comparison = fit_models(train, validation, config)
display(bundles['Logistic Regression']['pipeline'])
display(comparison[['train_seconds', 'threshold']].round(3))
display(Markdown(f'Three learned models plus the baseline fit in {comparison.train_seconds.sum():.2f} seconds. Numeric medians and categorical one-hot vocabularies are learned on training data; only logistic regression also scales numeric inputs.'))
'''), md('Logistic regression fits an additive decision rule, random forest averages small decision trees, and LightGBM adds trees that correct earlier errors. No hyperparameter search is used; validation chooses thresholds on the same fixed grid, requiring at least 85% precision where achievable.'), md('# 8. Metrics\nCompare models on the same validation period at their cost-minimising thresholds within the daily review budget and precision target, where achievable.'), code('''
display(comparison.round(4))
probabilities = {name: bundle['pipeline'].predict_proba(validation[FEATURES])[:, 1] for name, bundle in bundles.items()}
thresholds = {name: bundle['threshold'] for name, bundle in bundles.items()}
plots.pr_curves(validation.is_fraud, probabilities, 'Validation: compare precision across recall levels')
plots.confusion(validation.is_fraud, probabilities, thresholds)
plots.metric_bars(comparison)
best_ap = comparison.pr_auc.idxmax()
display(Markdown(f'{best_ap} has the highest validation PR-AUC ({comparison.loc[best_ap, "pr_auc"]:.3f}), against a fraud-rate reference of {validation.is_fraud.mean():.3f}. Threshold-dependent metrics use the same assumed costs and review budget.'))
'''), code('''
overfit = train_validation_scores(bundles, train, validation, config)
display(overfit.round(4))
gaps = overfit.train_pr_auc - overfit.validation_pr_auc
largest_gap = gaps.idxmax()
display(Markdown(f'{largest_gap} has the largest train-minus-validation PR-AUC gap ({gaps[largest_gap]:.3f}). A positive gap is a warning about overfitting or changed data, rather than proof of its cause.'))
'''), md('# 9. Choose model and threshold\nSelect the lowest validation cost among models meeting the quality gate and daily review budget, breaking cost ties by higher PR-AUC.'), code('''
selected = select_model(bundles, comparison, config)
name, threshold = selected['name'], selected['threshold']
plots.threshold_plot(selected['curve'], threshold)
row = comparison.loc[name]
peak = selected['curve'].loc[np.isclose(selected['curve'].threshold, threshold), 'peak_alerts_per_day'].iloc[0]
display(Markdown(f'Selected {name} at threshold {threshold:.2f} because it meets the quality gate with the lowest validation cost (ties use PR-AUC), INR {row.cost_per_txn:.2f} per transaction, with recall {row.recall:.2%} and precision {row.precision:.2%}. Its busiest validation day has {peak:.0f} alerts, within the assumed budget of {config["alert_budget_per_day"]} per day.'))
plots.importance_chart(importance(selected), name)
display(Markdown(f'The largest built-in importance belongs to {importance(selected).index[-1]}. Importance describes this fitted model, not a causal effect.'))
'''), md('# 10. Final test and next steps\nEvaluate the frozen model and threshold once on the held-out final 15 days to estimate performance on later transactions.'), code('''
test_probability = selected['pipeline'].predict_proba(test[FEATURES])[:, 1]
test_metrics = metrics(test.is_fraud, test_probability, threshold, config)
final = pd.DataFrame({'validation': selected['validation_metrics'], 'test': test_metrics}).T
display(final.round(4))
plots.confusion(test.is_fraud, {name: test_probability}, {name: threshold})
plots.pr_curves(test.is_fraud, {name: test_probability}, 'Final test: frozen model and threshold')
save_bundle(selected, ROOT / 'models' / 'candidate.joblib')
save_bundle(bundles['Always approve'], ROOT / 'models' / 'baseline.joblib')
display(Markdown(f'Test recall is {test_metrics["recall"]:.2%}, precision {test_metrics["precision"]:.2%}, PR-AUC {test_metrics["pr_auc"]:.3f}, and cost INR {test_metrics["cost_per_txn"]:.2f} per transaction. Test cost differs from validation by INR {test_metrics["cost_per_txn"]-row.cost_per_txn:+.2f}; no threshold is changed after this result.'))
'''), code('''
elapsed = time.perf_counter() - started
report = {'selected_model': name, 'threshold': threshold, 'data_version': config['data']['version'], 'comparison': comparison.reset_index().to_dict('records'), 'test': test_metrics, 'validation': selected['validation_metrics'], 'notebook_body_seconds': elapsed}
(ROOT / 'models' / 'ml_summary.json').write_text(json.dumps(report, indent=2))
display(Markdown(f'Saved the fitted pipeline and threshold in models/candidate.joblib; this notebook body ran in {elapsed:.2f} seconds on this machine. These results come from intentionally clear synthetic patterns; light label noise keeps the example imperfect.'))
'''), md('**Limitations and next steps**\n\n- Synthetic data: clear risk profiles and light label noise were designed for teaching; scores do not establish real-world performance.\n- More features: evaluate better transaction context on real, consented data.\n- Drift monitoring: check whether incoming data changes.\n- Delayed labels: account for when confirmed fraud becomes available.\n- Explainability: validate explanations with fraud reviewers.\n- Retraining: define when a new candidate should be trained and reviewed.')]
    write('01_fraud_detection.ipynb', cells)



def notebook_two():
    cells = [md('# 1. Git\nVersion code, configuration and the data generator so an experiment can be traced to the files that produced it.'), code("""
from pathlib import Path
import os
import sys
import time
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'src'))
started = time.perf_counter()
from IPython.display import display, Markdown, HTML
from fraud.git_demo import demonstrate
print(demonstrate())
"""), code("""
import subprocess
commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
display(Markdown(f'The local repository now includes a feature-branch edit, merge and tag at commit {commit}. Only demo_note changes, so the seed, model settings and costs stay fixed.'))
"""), md('# 2. Experiment tracking with MLflow\nRetrain the same three models and log parameters, validation metrics, pipelines, thresholds and provenance to local SQLite and artifact files.'), code("""
import pandas as pd
from fraud.config import load_config
from fraud.cleaning import clean_data, time_split
from fraud.features import add_features
from fraud.registry import track_models, start_ui
config = load_config()
frame = add_features(clean_data(pd.read_csv('data/transactions.csv')), config)
train, validation, _ = time_split(frame, config)
records, best, runs, comparison = track_models(train, validation, config)
columns = ['tags.mlflow.runName', 'metrics.pr_auc', 'metrics.recall', 'metrics.precision', 'metrics.cost_per_txn', 'params.threshold', 'params.data_version', 'tags.git_commit', 'tags.config_hash']
display(runs[columns])
display(Markdown(f'This experiment records {len(records)} learned models; {best} has the lowest validation cost at INR {comparison.loc[best, "cost_per_txn"]:.2f} per transaction. Git SHA and the configuration hash link each run to its inputs.'))
"""), code("""
url = start_ui()
display(Markdown(f'MLflow UI is responding at [{url}]({url}); experiment fraud-local contains the recorded runs. The database is mlflow.db and artifacts are under mlartifacts/.'))
"""), md('# 3. Model registry\nUse champion and challenger aliases to demonstrate a reversible model-version choice without deploying anything.'), code("""
import inspect
import mlflow
from mlflow import MlflowClient
from fraud.registry import register_candidates, alias_table, move_champion, load_champion
print(f'Installed MLflow: {mlflow.__version__}')
print(inspect.signature(MlflowClient.set_registered_model_alias))
print(inspect.signature(MlflowClient.get_model_version_by_alias))
client, original, challenger = register_candidates(records, best)
display(alias_table(client))
display(Markdown(f'The initial champion is version {original}; challenger is version {challenger}. Each registered version keeps its own threshold as a version tag.'))
"""), code("""
display(move_champion(client, challenger))
display(Markdown(f'Promotion moves champion to version {challenger}; challenger still points to that same version. This is an alias-mechanics demonstration, not a quality-approved release.'))
display(move_champion(client, original))
restored = load_champion()
display(Markdown(f'Rollback restores champion to version {restored["registry_version"]}, with estimator {restored["name"]} and threshold {restored["threshold"]:.2f}. The next section evaluates against this current alias.'))
"""), md('# 4. Tests and quality gate\nRun unit tests and compare each candidate with the current champion on validation data so acceptance has a repeatable rule.'), code("""!"{sys.executable}" -m pytest -q
assert get_ipython().user_ns['_exit_code'] == 0, 'pytest failed' """), code("""
import html
commands = [[sys.executable, 'scripts/run_gate.py'], [sys.executable, 'scripts/run_gate.py', '--candidate', 'models/baseline.joblib', '--output', 'models/blocked_summary.json']]
results = [subprocess.run(command, capture_output=True, text=True) for command in commands]
assert [result.returncode for result in results] == [0, 1]
columns = [f'<td style="vertical-align:top"><pre>{html.escape(result.stdout)}</pre><b>Exit {result.returncode}</b></td>' for result in results]
display(HTML('<table><tr><th>Selected model</th><th>Always-approve baseline</th></tr><tr>' + ''.join(columns) + '</tr></table>'))
display(Markdown(f'The selected model passes with exit {results[0].returncode}; the baseline is blocked with exit {results[1].returncode}. Tests cover deterministic data, row/batch preprocessing, known metric values, time boundaries and gate failures.'))
"""), md('# 5. CI with GitHub Actions\nWrite the workflow and run its checks locally so a commit can trigger the same tests and gate after a future push.\n\nThis workflow runs on GitHub’s servers after pushing; the local demonstration uses python scripts/run_ci_local.py and does not contact GitHub. A new CI runner uses an always-approve bootstrap reference because it has no historical registry; the full gate above compares with the actual local champion.')]
    cells += [md('Optional Docker and DVC checks also run in the hosted workflow: Docker on Linux and DVC on Linux and Windows. The local notebook runs the core checks; follow [the separate demo guide](../docs/docker_dvc_demo.md) for container startup and dataset restoration after this notebook.')]
    workflow = (ROOT / '.github/workflows/ci.yml').read_text()
    lines = workflow.splitlines()
    for offset in range(0, len(lines), 12):
        mode = '-a ' if offset else ''
        cells += [code(f'%%writefile {mode}.github/workflows/ci.yml\n' + '\n'.join(lines[offset:offset+12]))]
    cells += [code("""
print((ROOT / '.github/workflows/ci.yml').read_text())
ci = subprocess.run([sys.executable, 'scripts/run_ci_local.py'], capture_output=True, text=True)
print(ci.stdout)
if ci.returncode:
    print(ci.stderr)
ci.check_returncode()
from fraud.plots import setup, flow
setup()
flow()
display(Markdown(f'Local CI exits with code {ci.returncode}. CI tests the model and API and uploads a tiny release; registry aliases and local deployment remain explicit decisions.'))
"""), md('Local serving extends this workflow with a gated model release and a browser dashboard; run START-DEMO.cmd after the notebook to see requests reach FastAPI.'), code("""
from fraud.deployment import deploy
from fastapi.testclient import TestClient
from fraud.api import create_app
release = deploy()
with TestClient(create_app()) as api:
    example = api.get('/api/samples?mode=fraud').json()['samples'][0]
    prediction = api.post('/predict', json=example['transaction'])
    prediction.raise_for_status()
    display(prediction.json())
display(Markdown(f'Local release {release["release"]} serves {release["model"]}; the example returns {prediction.json()["decision"]}. The synthetic label is kept outside the prediction request.'))
"""), code("""
import json
elapsed = time.perf_counter() - started
report = {'notebook_body_seconds': elapsed, 'gate_exit_codes': [result.returncode for result in results], 'ci_exit_code': ci.returncode, 'aliases': alias_table(client).to_dict('records')}
(ROOT / 'models/mlops_summary.json').write_text(json.dumps(report, indent=2))
display(Markdown(f'The MLOps notebook body ran in {elapsed:.2f} seconds on this machine, including {len(records)} tracked models, alias promotion and rollback, tests and both gate outcomes. These are local notebook checks; the repository Actions page records hosted Linux and Windows results.'))
""")]
    write('02_mlops.ipynb', cells)


if __name__ == '__main__':
    notebook_one()
    notebook_two()
