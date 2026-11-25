"""Create a quality-gated local release, or roll back to the previous release."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.config import ROOT
from fraud.deployment import deploy, rollback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', default='models/candidate.joblib')
    parser.add_argument('--data', default='data/transactions.csv')
    parser.add_argument('--directory', default='models/deployed')
    parser.add_argument('--baseline', default='models/baseline.joblib')
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    try:
        manifest = (rollback(ROOT / args.directory) if args.rollback else
                    deploy(ROOT / args.candidate, ROOT / args.data,
                           ROOT / args.directory, ROOT / args.baseline))
        print(f'PASS: active local release {manifest["release"]}: {manifest["model"]}, '
              f'threshold {manifest["threshold"]:.2f}')
        print('If the API is already running, click Reload model in the dashboard.')
    except (OSError, ValueError, KeyError) as error:
        print(f'BLOCK: {error}')
        raise SystemExit(1)
