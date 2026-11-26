"""Stage an existing, checked release and replay data for the Docker build."""
import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.config import ROOT
from fraud.serving import Replay, load_release


def prepare(directory, data_path):
    predictor = load_release(directory)  # Validate checksum, version and feature settings.
    Replay(data_path)  # Fail here if the replay dataset cannot be loaded.
    staging = ROOT / '.container'
    if staging.exists():
        shutil.rmtree(staging)
    destination = staging / 'models/deployed'
    destination.mkdir(parents=True)
    (staging / 'data').mkdir()
    shutil.copyfile(directory / 'active.json', destination / 'active.json')
    name = predictor.manifest['artifact']
    shutil.copyfile(directory / name, destination / name)
    shutil.copyfile(data_path, staging / 'data/transactions.csv')
    print(f'Staged {predictor.manifest["release"]}: {predictor.bundle["name"]}; '
          f'threshold {predictor.bundle["threshold"]:.2f}. No training performed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='models/deployed')
    parser.add_argument('--data', default='data/transactions.csv')
    args = parser.parse_args()
    prepare(ROOT / args.directory, ROOT / args.data)
