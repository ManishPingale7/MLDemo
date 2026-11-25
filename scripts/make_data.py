"""Create the full dataset, or a separate tiny CI dataset."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.config import ROOT, load_config
from fraud.data import generate_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', choices=['full', 'tiny'], default='full')
    args = parser.parse_args()
    frame = generate_data(load_config(), args.preset)
    path = ROOT / 'data' / ('transactions.csv' if args.preset == 'full' else 'tiny.csv')
    path.parent.mkdir(exist_ok=True)
    frame.to_csv(path, index=False)
    print(f'{path.name}: {len(frame):,} rows; fraud {frame.is_fraud.mean():.3%}')
