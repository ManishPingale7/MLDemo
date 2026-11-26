"""Check a running container through real HTTP, using only the standard library."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def check(base_url, output):
    def request(path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = Request(base_url + path, data=data, headers={'Content-Type': 'application/json'})
        with urlopen(req, timeout=5) as response:
            return response.read()

    deadline = time.monotonic() + 45
    while True:
        try:
            health = json.loads(request('/health'))
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise RuntimeError('Container did not become ready within 45 seconds.') from None
            time.sleep(0.5)
    assert health['status'] == 'ready'
    for path in ['/', '/docs', '/assets/app.js', '/assets/style.css']:
        assert request(path)
    samples = json.loads(request('/api/samples?mode=balanced&count=6'))['samples']
    predictions = []
    for sample in samples:
        result = json.loads(request('/predict', sample['transaction']))
        assert 0 <= result['score'] <= 1
        assert result['decision'] == ('review' if result['score'] >= result['threshold'] else 'approve')
        assert result['release'] == health['release']
        predictions.append(result)
    try:
        request('/predict', {**samples[0]['transaction'], 'is_fraud': 1})
    except HTTPError as error:
        assert error.code == 422
    else:
        raise AssertionError('API accepted a label as a feature.')
    assert json.loads(request('/api/reload', {})) == health
    report = {'passed': True, 'health': health, 'predictions': predictions,
              'label_input_rejected': True, 'reload_passed': True}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'PASS container: {len(predictions)} real HTTP predictions, static pages, '
          'label rejection and model reload.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path, default=Path('models/container_smoke.json'))
    args = parser.parse_args()
    check(args.url.rstrip('/'), args.output)
