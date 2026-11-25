"""Tiny CI smoke training, isolated from the full dataset and registry."""
import json
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.config import ROOT, load_config
from fraud.cleaning import clean_data, time_split
from fraud.features import add_features
from fraud.models import fit_models, save_bundle, select_model

if __name__ == '__main__':
    config = load_config()
    frame = add_features(clean_data(pd.read_csv(ROOT / 'data/tiny.csv')), config)
    train, validation, _ = time_split(frame, config)
    scale = config['presets']['tiny']['rows'] / config['presets']['full']['rows']
    bundles, comparison = fit_models(train, validation, config, scale)
    selected = select_model(bundles, comparison)
    save_bundle(selected, ROOT / 'models/ci/candidate.joblib')
    # Fresh CI has no historical registry: bootstrap its reference with the baseline.
    save_bundle(bundles['Always approve'], ROOT / 'models/ci/reference.joblib')
    (ROOT / 'models/ci/comparison.json').write_text(json.dumps(comparison.reset_index().to_dict('records'), indent=2))
    print(comparison[['precision', 'recall', 'pr_auc', 'cost_per_txn']].round(4).to_string())
    print(f'Tiny candidate: {selected["name"]}, threshold {selected["threshold"]:.2f}')
