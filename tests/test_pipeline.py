import numpy as np
import pytest
from fraud.cleaning import clean_data, time_split
from fraud.config import load_config
from fraud.data import generate_data
from fraud.features import FEATURES, add_features
from fraud.models import make_pipeline
from threadpoolctl import threadpool_limits


@pytest.fixture(scope='module')
def parts():
    config = load_config()
    return time_split(add_features(clean_data(generate_data(config, 'tiny')), config), config)


def test_time_split(parts):
    train, validation, test = parts
    assert train.timestamp.max() < validation.timestamp.min()
    assert validation.timestamp.max() < test.timestamp.min()
    assert len(set(train.txn_id) & set(test.txn_id)) == 0


@pytest.mark.parametrize('name', ['Logistic Regression', 'Random Forest', 'LightGBM'])
def test_batch_invariance(parts, name):
    train, validation, _ = parts
    pipeline = make_pipeline(name, load_config(), 60)
    with threadpool_limits(limits=1):
        pipeline.fit(train[FEATURES], train.is_fraud)
    batch = validation[FEATURES].iloc[:10].copy()
    batch.iloc[0, batch.columns.get_loc('distance_from_home_km')] = np.nan
    prep = pipeline.named_steps['preprocess']
    np.testing.assert_allclose(prep.transform(batch.iloc[:1]), prep.transform(batch)[:1])
    np.testing.assert_allclose(pipeline.predict_proba(batch.iloc[:1]),
                               pipeline.predict_proba(batch)[:1])
    expected = train.distance_from_home_km.median()
    actual = prep.named_transformers_['numeric'].named_steps['impute'].statistics_[1]
    assert actual == expected
