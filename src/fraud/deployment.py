"""Manual local releases: validate first, then atomically change active.json."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import joblib
import pandas as pd

from fraud.cleaning import clean_data, time_split
from fraud.config import ROOT, load_config
from fraud.features import add_features
from fraud.gate import check_gate, evaluate_bundle


def deploy(candidate_path=None, data_path=None, directory=None, baseline_path=None):
    config = load_config()
    candidate_path = Path(candidate_path or ROOT / 'models/candidate.joblib')
    directory = Path(directory or ROOT / 'models/deployed')
    baseline_path = Path(baseline_path or ROOT / 'models/baseline.joblib')
    directory.mkdir(parents=True, exist_ok=True)
    active_path = directory / 'active.json'
    previous = json.loads(active_path.read_text()) if active_path.exists() else None
    reference_path = directory / previous['artifact'] if previous else baseline_path
    candidate, reference = joblib.load(candidate_path), joblib.load(reference_path)
    for bundle in [candidate, reference]:
        if bundle.get('data_version') != config['data']['version']:
            raise ValueError('Model and dataset versions differ; rerun the notebooks.')
    frame = add_features(clean_data(pd.read_csv(data_path or ROOT / 'data/transactions.csv')), config)
    _, validation, _ = time_split(frame, config)
    scores = evaluate_bundle(candidate, validation, config)
    reference_scores = evaluate_bundle(reference, validation, config)
    passed, reasons = check_gate(scores, reference_scores, config)
    if not passed:
        raise ValueError('Quality gate blocked release: ' + '; '.join(reasons))
    number = max([int(p.stem.split('-')[1]) for p in directory.glob('release-*.joblib')] or [0]) + 1
    release = f'release-{number:03d}'
    artifact = directory / f'{release}.joblib'
    shutil.copyfile(candidate_path, artifact)
    manifest = {'release': release, 'artifact': artifact.name,
                'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                'model': candidate['name'], 'threshold': candidate['threshold'],
                'data_version': candidate['data_version'], 'feature_config': config['features'],
                'created_at': datetime.now(timezone.utc).isoformat(),
                'validation': scores, 'gate_passed': True,
                'reference': previous['release'] if previous else 'always-approve bootstrap',
                'previous': previous['release'] if previous else None}
    (directory / f'{release}.json').write_text(json.dumps(manifest, indent=2))
    temporary = directory / 'active.tmp'
    temporary.write_text(json.dumps(manifest, indent=2))
    os.replace(temporary, active_path)
    return manifest


def rollback(directory=None):
    directory = Path(directory or ROOT / 'models/deployed')
    active = json.loads((directory / 'active.json').read_text())
    if not active['previous']:
        raise ValueError('There is no previous local release to restore.')
    previous = json.loads((directory / f"{active['previous']}.json").read_text())
    temporary = directory / 'active.tmp'
    temporary.write_text(json.dumps(previous, indent=2))
    os.replace(temporary, directory / 'active.json')
    return previous
