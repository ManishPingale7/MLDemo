"""Train a full-data release candidate without executing either notebook."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.cleaning import clean_data, time_split
from fraud.config import ROOT, load_config
from fraud.features import add_features
from fraud.models import fit_models, save_bundle, select_model


def main():
    config = load_config()
    data = pd.read_csv(ROOT / 'data/transactions.csv')
    frame = add_features(clean_data(data), config)
    train, validation, _ = time_split(frame, config)
    # Selection uses validation only; the final test period is not evaluated here.
    bundles, comparison = fit_models(train, validation, config)
    selected = select_model(bundles, comparison, config)
    destination = ROOT / 'models/publish'
    save_bundle(selected, destination / 'candidate.joblib')
    # Hosted builds have no previous production champion: use an explicit bootstrap baseline.
    save_bundle(bundles['Always approve'], destination / 'baseline.joblib')
    print(comparison[['precision', 'recall', 'pr_auc', 'cost_per_txn']].round(4).to_string())
    print(f'Full-data candidate: {selected["name"]}; threshold {selected["threshold"]:.2f}')


if __name__ == '__main__':
    main()
