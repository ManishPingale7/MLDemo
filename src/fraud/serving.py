"""Load one local release and reuse the notebook's exact feature pipeline."""
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from fraud.cleaning import clean_data, time_split
from fraud.config import ROOT, load_config
from fraud.features import FEATURES, add_features

RAW_FEATURES = ['txn_id', 'timestamp', 'customer_id', 'amount_inr', 'merchant_category',
                'city', 'distance_from_home_km', 'device_type', 'is_new_merchant',
                'txns_last_1h', 'customer_avg_amount_inr']


class Predictor:
    """Immutable model snapshot: one request always uses one release."""

    def __init__(self, bundle, manifest, config):
        if bundle.get('data_version') != config['data']['version']:
            raise ValueError('Model and configured dataset versions differ.')
        self.bundle = bundle
        self.manifest = manifest
        self.config = config

    def predict(self, transaction):
        start = perf_counter()
        frame = add_features(clean_data(pd.DataFrame([transaction])), self.config)
        score = float(self.bundle['pipeline'].predict_proba(frame[FEATURES])[0, 1])
        threshold = float(self.bundle['threshold'])
        return {'txn_id': transaction['txn_id'], 'score': score, 'threshold': threshold,
                'decision': 'review' if score >= threshold else 'approve',
                'model': self.bundle['name'], 'release': self.manifest['release'],
                'data_version': self.bundle['data_version'],
                'latency_ms': round((perf_counter() - start) * 1000, 2)}


def load_release(directory=None, config=None):
    directory = Path(directory or ROOT / 'models/deployed')
    config = config or load_config()
    manifest = json.loads((directory / 'active.json').read_text())
    path = directory / manifest['artifact']
    if path.resolve().parent != directory.resolve():
        raise ValueError('Release artifact must be inside its release directory.')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('Release checksum does not match the model file.')
    if manifest['feature_config'] != config['features']:
        raise ValueError('Feature configuration changed; create a new release.')
    return Predictor(joblib.load(path), manifest, config)


class Replay:
    """Deterministic validation-only replay; labels never enter the predictor."""

    def __init__(self, data_path=None, config=None):
        self.config = config or load_config()
        raw = pd.read_csv(data_path or ROOT / 'data/transactions.csv')
        _, validation, _ = time_split(clean_data(raw), self.config)
        order = np.random.default_rng(self.config['seed']).permutation(len(validation))
        self.frame = validation.iloc[order].reset_index(drop=True)
        self.routine = self.frame.loc[self.frame.is_fraud == 0].reset_index(drop=True)
        self.fraud = self.frame.loc[self.frame.is_fraud == 1].reset_index(drop=True)
        if self.routine.empty or self.fraud.empty:
            raise ValueError('Replay requires both labels in the validation period.')

    def sample(self, mode, offset):
        if mode == 'natural':
            row = self.frame.iloc[offset % len(self.frame)]
        else:
            use_fraud = mode == 'fraud' or (mode == 'balanced' and offset % 2 == 1)
            pool = self.fraud if use_fraud else self.routine
            index = offset // 2 if mode == 'balanced' else offset
            row = pool.iloc[index % len(pool)]
        # JSON conversion makes missing distance null and timestamps ISO strings.
        values = json.loads(row[RAW_FEATURES].to_json(date_format='iso'))
        return {'transaction': values, 'actual_fraud': bool(row.is_fraud),
                'source': 'synthetic validation replay', 'position': offset}

    def info(self):
        return {'rows': len(self.frame), 'fraud_rate': float(self.frame.is_fraud.mean()),
                'start': str(self.frame.timestamp.min()), 'end': str(self.frame.timestamp.max())}
