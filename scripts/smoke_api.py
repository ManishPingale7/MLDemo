"""Exercise the deployed API through HTTP-style requests without opening a browser."""
import argparse
import json
from pathlib import Path
import sys
from fastapi.testclient import TestClient
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fraud.api import create_app
from fraud.config import ROOT


def smoke(directory, data_path, output):
    with TestClient(create_app(directory, data_path)) as client:
        health = client.get('/health')
        assert health.status_code == 200
        assert client.get('/').status_code == 200
        assert client.get('/docs').status_code == 200
        samples = client.get('/api/samples?mode=balanced&count=6').json()['samples']
        predictions = []
        for sample in samples:
            response = client.post('/predict', json=sample['transaction'])
            assert response.status_code == 200, response.text
            result = response.json()
            assert 0 <= result['score'] <= 1
            assert result['decision'] == ('review' if result['score'] >= result['threshold'] else 'approve')
            predictions.append(result)
        leaking = {**samples[0]['transaction'], 'is_fraud': 1}
        assert client.post('/predict', json=leaking).status_code == 422
        assert client.post('/api/reload').status_code == 200
    report = {'passed': True, 'health': health.json(), 'predictions': predictions,
              'label_input_rejected': True, 'reload_passed': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f'PASS API smoke: {len(predictions)} requests, label rejection, reload and offline pages.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='models/deployed')
    parser.add_argument('--data', default='data/transactions.csv')
    parser.add_argument('--output', default='models/api_smoke.json')
    args = parser.parse_args()
    smoke(ROOT / args.directory, ROOT / args.data, ROOT / args.output)
