"""Local MLflow tracking and alias-based registry, verified against MLflow 3.6.0."""
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import urllib.request

os.environ.setdefault('MLFLOW_ENABLE_TELEMETRY', 'false')
os.environ.setdefault('DO_NOT_TRACK', 'true')
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from fraud.config import ROOT, config_hash
from fraud.features import FEATURES, NUMERIC
from fraud.models import fit_models, select_model

MODEL_NAME = 'fraud-classroom'


def setup_tracking():
    # A relative SQLite URI resolves under the project working directory.
    os.chdir(ROOT)
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    client = MlflowClient()
    experiment = client.get_experiment_by_name('fraud-local')
    if experiment is None:
        experiment_id = client.create_experiment('fraud-local',
                                                 artifact_location=(ROOT / 'mlartifacts').as_uri())
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)
    return client, experiment_id


def track_models(train, validation, config):
    client, experiment_id = setup_tracking()
    bundles, comparison = fit_models(train, validation, config)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    requirements = [f'{name}=={importlib.metadata.version(name)}' for name in
                    ['mlflow', 'scikit-learn', 'lightgbm', 'numpy', 'pandas', 'scipy',
                     'cloudpickle', 'joblib']]
    records = {}
    for name, bundle in bundles.items():
        if name == 'Always approve':
            continue
        with mlflow.start_run(run_name=name) as run:
            estimator = bundle['pipeline'].named_steps['model']
            parameters = {key: value for key, value in estimator.get_params().items()
                          if isinstance(value, (str, int, float, bool)) or value is None}
            mlflow.log_params({**parameters, 'seed': config['seed'], 'data_version': config['data']['version'],
                               'threshold': bundle['threshold'], 'config_hash': config_hash(),
                               'missed_fraud_cost': config['costs']['missed_fraud'],
                               'false_alarm_cost': config['costs']['false_alarm']})
            mlflow.set_tags({'git_commit': commit, 'config_hash': config_hash()})
            mlflow.log_metrics(bundle['validation_metrics'])
            mlflow.log_metric('train_seconds', comparison.loc[name, 'train_seconds'])
            example = validation[FEATURES].head(3).astype({column: float for column in NUMERIC})
            signature = infer_signature(example, bundle['pipeline'].predict_proba(example))
            info = mlflow.sklearn.log_model(bundle['pipeline'], name='pipeline',
                                            signature=signature, pip_requirements=requirements,
                                            pyfunc_predict_fn='predict_proba')
            mlflow.log_dict({'threshold': bundle['threshold'], 'model': name}, 'threshold.json')
            mlflow.log_artifact(str(ROOT / 'config.yaml'))
            records[name] = {'run_id': run.info.run_id, 'uri': info.model_uri,
                             'threshold': bundle['threshold'], 'data_version': config['data']['version']}
    best = select_model(bundles, comparison, config)['name']
    runs = mlflow.search_runs(experiment_ids=[experiment_id],
                              filter_string=f"tags.git_commit = '{commit}'")
    return records, best, runs, comparison


def alias_table(client):
    rows = []
    for alias in ['champion', 'challenger']:
        version = client.get_model_version_by_alias(MODEL_NAME, alias)
        rows.append({'alias': alias, 'version': version.version, 'run_id': version.run_id})
    return pd.DataFrame(rows)


def register_candidates(records, best):
    client, _ = setup_tracking()
    candidate = next(name for name in records if name != best)
    winner = mlflow.register_model(records[best]['uri'], MODEL_NAME)
    other = mlflow.register_model(records[candidate]['uri'], MODEL_NAME)
    for version, name in [(winner, best), (other, candidate)]:
        client.set_model_version_tag(MODEL_NAME, version.version, 'threshold',
                                     str(records[name]['threshold']))
        client.set_model_version_tag(MODEL_NAME, version.version, 'estimator', name)
        client.set_model_version_tag(MODEL_NAME, version.version, 'data_version', records[name]['data_version'])
    client.set_registered_model_alias(MODEL_NAME, 'champion', winner.version)
    client.set_registered_model_alias(MODEL_NAME, 'challenger', other.version)
    return client, winner.version, other.version


def move_champion(client, version):
    client.set_registered_model_alias(MODEL_NAME, 'champion', version)
    return alias_table(client)


def load_champion():
    client, _ = setup_tracking()
    version = client.get_model_version_by_alias(MODEL_NAME, 'champion')
    pipeline = mlflow.sklearn.load_model(f'models:/{MODEL_NAME}/{version.version}')
    return {'pipeline': pipeline, 'threshold': float(version.tags['threshold']),
            'name': version.tags['estimator'], 'registry_version': version.version,
            'data_version': version.tags.get('data_version')}


def start_ui():
    """Start a loopback-only UI, or reuse this project's existing healthy UI."""
    url = 'http://127.0.0.1:5000'
    marker = ROOT / '.runtime' / 'mlflow_ui.json'
    try:
        with urllib.request.urlopen(url + '/health', timeout=1) as response:
            healthy = response.status == 200
        if healthy and marker.exists():
            return url
        raise RuntimeError('Port 5000 is already occupied; stop that service and rerun.')
    except (OSError, urllib.error.URLError):
        pass
    log_path = ROOT / 'mlflow-ui.log'
    detached = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
                if os.name == 'nt' else {'start_new_session': True})
    with log_path.open('a') as log:
        process = subprocess.Popen([sys.executable, '-m', 'mlflow', 'ui',
                                    '--backend-store-uri', 'sqlite:///mlflow.db',
                                    '--host', '127.0.0.1', '--port', '5000', '--workers', '1'],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, **detached)
    for _ in range(80):
        if process.poll() is not None:
            raise RuntimeError(f'MLflow UI exited; see {log_path}')
        try:
            with urllib.request.urlopen(url + '/health', timeout=0.5) as response:
                if response.status == 200:
                    marker.parent.mkdir(exist_ok=True)
                    marker.write_text(json.dumps({'pid': process.pid, 'url': url}))
                    return url
        except (OSError, urllib.error.URLError):
            time.sleep(0.1)
    raise RuntimeError(f'MLflow UI did not become ready; see {log_path}')
