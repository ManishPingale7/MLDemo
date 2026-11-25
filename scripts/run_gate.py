"""Exit 0 for acceptance and 1 for rejection, including missing/invalid artifacts."""
import argparse
import json
from pathlib import Path
import sys
import joblib
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.config import ROOT, load_config
from fraud.cleaning import clean_data, time_split
from fraud.features import add_features
from fraud.gate import check_gate, evaluate_bundle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='models/candidate.joblib')
    parser.add_argument('--champion', help='Explicit local reference; default is current MLflow champion alias')
    parser.add_argument('--data', default='data/transactions.csv')
    parser.add_argument('--output', default='models/gate_summary.json')
    args = parser.parse_args()
    try:
        config = load_config()
        candidate = joblib.load(ROOT / args.candidate)
        if args.champion:
            champion = joblib.load(ROOT / args.champion)
        else:
            from fraud.registry import load_champion
            champion = load_champion()
        frame = add_features(clean_data(pd.read_csv(ROOT / args.data)), config)
        _, validation, _ = time_split(frame, config)
        scores = evaluate_bundle(candidate, validation, config)
        reference = evaluate_bundle(champion, validation, config)
        passed, reasons = check_gate(scores, reference, config)
        result = {'passed': passed, 'candidate': candidate['name'], 'candidate_metrics': scores,
                  'champion': champion['name'], 'champion_metrics': reference, 'reasons': reasons}
        path = ROOT / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2))
        print(('PASS' if passed else 'BLOCK') + ': ' + '; '.join(reasons))
        print(f'PR-AUC={scores["pr_auc"]:.4f}; recall={scores["recall"]:.4f}; '
              f'cost=INR {scores["cost_per_txn"]:.2f}/transaction')
        return 0 if passed else 1
    except Exception as error:
        print(f'BLOCK: {type(error).__name__}: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
