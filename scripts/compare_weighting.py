"""Compare three LightGBM class weights on validation; never score the test set."""
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.cleaning import clean_data, time_split
from fraud.config import ROOT, load_config
from fraud.evaluate import best_threshold, metrics
from fraud.features import FEATURES, add_features
from fraud.models import make_pipeline


def main():
    config = load_config()
    frame = add_features(clean_data(pd.read_csv(ROOT / 'data/transactions.csv')), config)
    train, validation, _ = time_split(frame, config)
    original_weight = (train.is_fraud == 0).sum() / train.is_fraud.sum()
    settings = [('Original', original_weight), ('Moderate', np.sqrt(original_weight)),
                ('Equal', 1.0)]
    comparison, tradeoffs = [], []
    for label, weight in settings:
        model = make_pipeline('LightGBM', config, weight)
        start = perf_counter()
        with threadpool_limits(limits=1):
            model.fit(train[FEATURES], train.is_fraud)
        probability = model.predict_proba(validation[FEATURES])[:, 1]
        threshold, curve = best_threshold(validation.is_fraud, probability, config,
                                           validation.timestamp.dt.date)
        scores = metrics(validation.is_fraud, probability, threshold, config)
        comparison.append({'setting': label, 'weight': float(weight), 'threshold': threshold,
                           **scores, 'seconds': perf_counter() - start})
        for minimum_recall in [0.65, 0.60, 0.50]:
            eligible = curve.loc[curve.within_budget & (curve.recall >= minimum_recall)]
            if eligible.empty:
                continue
            chosen = eligible.sort_values(['precision', 'recall'], ascending=False).iloc[0]
            tradeoffs.append({'setting': label, 'minimum_recall': minimum_recall,
                              **chosen.to_dict()})
    report = {'evaluation_split': 'validation only', 'cost_selected': comparison,
              'precision_focused': tradeoffs,
              'note': 'Exploratory comparison; original candidate and test report are unchanged.'}
    output = ROOT / 'models/weighting_comparison.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    columns = ['setting', 'threshold', 'precision', 'recall', 'pr_auc', 'cost_per_txn', 'alerts']
    print('Same validation data, costs and daily alert budget:')
    print(pd.DataFrame(comparison)[columns].round(4).to_string(index=False))
    print('\nTradeoffs when prioritising precision at a minimum recall:')
    print(pd.DataFrame(tradeoffs).round(4).to_string(index=False))
    print(f'\nSaved {output.name}. Test set was not scored; original artifacts are unchanged.')


if __name__ == '__main__':
    main()
