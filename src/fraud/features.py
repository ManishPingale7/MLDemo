"""Four row-local features; none learn from future rows or labels."""
import numpy as np

BASE_NUMERIC = ['amount_inr', 'distance_from_home_km', 'is_new_merchant', 'txns_last_1h',
                'customer_avg_amount_inr', 'hour', 'day_of_week']
ENGINEERED = ['is_night', 'amount_to_avg_ratio', 'is_far_from_home', 'log_amount']
NUMERIC = BASE_NUMERIC + ENGINEERED
CATEGORICAL = ['merchant_category', 'city', 'device_type']
FEATURES = NUMERIC + CATEGORICAL


def add_features(frame, config):
    frame = frame.copy()
    frame['is_night'] = (frame.hour < 6).astype(int)
    frame['amount_to_avg_ratio'] = frame.amount_inr / frame.customer_avg_amount_inr
    distance = frame.distance_from_home_km
    frame['is_far_from_home'] = np.where(distance.isna(), np.nan,
                                        (distance > config['features']['far_km']).astype(float))
    frame['log_amount'] = np.log1p(frame.amount_inr)
    return frame
