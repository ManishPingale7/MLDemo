import json
from urllib.error import HTTPError

from scripts import smoke_container


def test_check_retries_connection_reset_while_waiting_for_health(monkeypatch, tmp_path):
    health = {'status': 'ready', 'release': 'release-1'}
    calls = []

    def urlopen(request, timeout):
        path = request.full_url.rsplit('/', 1)[-1]
        calls.append(path)
        if path == 'health' and calls.count('health') == 1:
            raise ConnectionResetError('connection reset by peer')
        if path == 'health' or path == 'reload':
            return _Response(health)
        if path == 'samples?mode=balanced&count=6':
            return _Response({'samples': [{'transaction': {}}]})
        if path == 'predict':
            if b'is_fraud' in request.data:
                raise HTTPError(request.full_url, 422, 'Unprocessable Entity', None, None)
            return _Response({'score': 0.5, 'threshold': 0.5,
                              'decision': 'review', 'release': 'release-1'})
        return _Response({})

    monkeypatch.setattr(smoke_container, 'urlopen', urlopen)
    monkeypatch.setattr(smoke_container.time, 'sleep', lambda _: None)

    output = tmp_path / 'smoke.json'
    smoke_container.check('http://127.0.0.1:8765', output)

    assert calls.count('health') == 2
    assert json.loads(output.read_text())['passed'] is True


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value).encode()
