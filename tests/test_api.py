"""Real fitted pipelines, local releases and HTTP requests; no cloud or saved files required."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.api import create_app
from fraud.cleaning import clean_data, time_split
from fraud.config import load_config
from fraud.data import generate_data
from fraud.deployment import deploy, rollback
from fraud.features import FEATURES, add_features
from fraud.models import fit_models, save_bundle, select_model
from fraud.serving import Replay, load_release


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    directory = tmp_path_factory.mktemp('serving')
    config = load_config()
    raw = generate_data(config, 'tiny')
    data = directory / 'tiny.csv'
    raw.to_csv(data, index=False)
    train, validation, _ = time_split(add_features(clean_data(raw), config), config)
    bundles, comparison = fit_models(train, validation, config, scale=0.1)
    selected = select_model(bundles, comparison, config)
    candidate, baseline = directory / 'candidate.joblib', directory / 'baseline.joblib'
    save_bundle(selected, candidate)
    save_bundle(bundles['Always approve'], baseline)
    releases = directory / 'releases'
    deploy(candidate, data, releases, baseline)
    return {'config': config, 'data': data, 'selected': selected,
            'candidate': candidate, 'baseline': baseline, 'releases': releases}


@pytest.fixture()
def client(setup):
    with TestClient(create_app(setup['releases'], setup['data'])) as value:
        yield value


@pytest.fixture()
def transaction(client):
    return client.get('/api/samples?mode=fraud&offset=0').json()['samples'][0]['transaction']


def test_prediction_matches_saved_pipeline(client, transaction, setup):
    response = client.post('/predict', json=transaction)
    assert response.status_code == 200
    result = response.json()
    import pandas as pd
    frame = add_features(clean_data(pd.DataFrame([transaction])), setup['config'])
    expected = setup['selected']['pipeline'].predict_proba(frame[FEATURES])[0, 1]
    assert result['score'] == pytest.approx(expected)
    assert result['decision'] == ('review' if expected >= result['threshold'] else 'approve')
    assert result['data_version'] == 'teaching-v2'
    assert result['latency_ms'] >= 0


@pytest.mark.parametrize('field,value', [('amount_inr', -1), ('amount_inr', 0),
                                       ('customer_avg_amount_inr', 0),
                                       ('amount_inr', 'Infinity'),
                                       ('distance_from_home_km', -5),
                                       ('device_type', 'unrecognised'),
                                       ('is_new_merchant', 2), ('txns_last_1h', -1),
                                       ('timestamp', '2025-03-10T12:00:00Z'),
                                       ('is_fraud', 1)])
def test_invalid_or_leaking_inputs_rejected(client, transaction, field, value):
    transaction[field] = value
    assert client.post('/predict', json=transaction).status_code == 422


def test_missing_values_and_category_normalisation(client, transaction):
    transaction.update(distance_from_home_km=None, device_type=None,
                       merchant_category=' ELECTRONICS ')
    response = client.post('/predict', json=transaction)
    assert response.status_code == 200
    assert np.isfinite(response.json()['score'])


def test_replay_is_deterministic_and_validation_only(client, setup):
    first = client.get('/api/samples?mode=balanced&count=10').json()
    assert first == client.get('/api/samples?mode=balanced&count=10').json()
    assert sum(item['actual_fraud'] for item in first['samples']) == 5
    replay = Replay(setup['data'], setup['config'])
    train_end = replay.frame.timestamp.min().normalize()
    assert str(train_end.date()) >= '2025-03-02'
    assert str(replay.frame.timestamp.max().date()) < '2025-03-17'
    assert all('is_fraud' not in item['transaction'] for item in first['samples'])
    assert client.get('/api/samples?mode=oops').status_code == 422
    assert client.get('/api/samples?count=10000').status_code == 422
    assert client.get('/api/samples?offset=-1').status_code == 422


def test_frontend_and_offline_api_reference(client):
    for path in ['/', '/assets/style.css', '/assets/app.js', '/assets/icon.svg',
                 '/docs', '/openapi.json', '/health', '/api/model']:
        assert client.get(path).status_code == 200
    assert 'cdn.' not in client.get('/docs').text
    assert 'is_fraud' not in client.get('/openapi.json').json()['components']['schemas']['Transaction']['properties']


def test_gate_blocks_bad_release_and_keeps_active(setup):
    active_path = setup['releases'] / 'active.json'
    previous = active_path.read_bytes()
    with pytest.raises(ValueError, match='Quality gate blocked'):
        deploy(setup['baseline'], setup['data'], setup['releases'], setup['baseline'])
    assert active_path.read_bytes() == previous


def test_reload_and_rollback(client, setup, transaction):
    before = client.get('/health').json()['release']
    manifest = deploy(setup['candidate'], setup['data'], setup['releases'], setup['baseline'])
    assert client.get('/health').json()['release'] == before  # explicit reload required
    assert client.post('/api/reload').json()['release'] == manifest['release']
    assert client.post('/predict', json=transaction).json()['release'] == manifest['release']
    assert rollback(setup['releases'])['release'] == before
    assert client.post('/api/reload').json()['release'] == before


def test_corrupt_release_is_not_loaded(client, setup, transaction):
    active_path = setup['releases'] / 'active.json'
    original = active_path.read_text()
    manifest = json.loads(original)
    manifest['sha256'] = 'incorrect'
    active_path.write_text(json.dumps(manifest))
    try:
        assert client.post('/api/reload').status_code == 409
        assert client.post('/predict', json=transaction).status_code == 200
        with pytest.raises(ValueError, match='checksum'):
            load_release(setup['releases'])
    finally:
        active_path.write_text(original)
