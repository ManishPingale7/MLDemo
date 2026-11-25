import pandas as pd
import pytest
from fraud.config import load_config
from fraud.data import COLUMNS, generate_data


@pytest.mark.parametrize('preset', ['tiny', 'full'])
def test_data(preset):
    config = load_config()
    frame = generate_data(config, preset)
    pd.testing.assert_frame_equal(frame, generate_data(config, preset))
    assert list(frame) == COLUMNS
    assert 0.01 <= frame.is_fraud.mean() <= 0.02
    assert 0.023 < frame.distance_from_home_km.isna().mean() < 0.027
    assert 0.018 < frame.device_type.isna().mean() < 0.022
    assert 0.0025 < frame.duplicated().mean() < 0.0035
    assert (frame.amount_inr <= 0).sum() >= 4
    assert frame.merchant_category.str.contains(' ').any()
    assert frame.drop_duplicates().txn_id.is_unique
