"""Deterministic row cleanup; learned imputation belongs in the training pipeline."""
import pandas as pd
from fraud.config import load_config


def clean_data(raw):
    frame = raw.drop_duplicates().copy()
    frame = frame.loc[frame.amount_inr > 0].copy()
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    frame['merchant_category'] = frame.merchant_category.str.strip().str.lower()
    frame['device_type'] = frame.device_type.fillna('unknown')
    frame['hour'] = frame.timestamp.dt.hour
    frame['day_of_week'] = frame.timestamp.dt.dayofweek
    return frame.sort_values(['timestamp', 'txn_id']).reset_index(drop=True)


def time_split(frame, config=None):
    config = config or load_config()
    start = pd.Timestamp(config['data']['start'])
    train_end = start + pd.Timedelta(days=60)
    val_end = start + pd.Timedelta(days=75)
    return (frame.loc[frame.timestamp < train_end].copy(),
            frame.loc[(frame.timestamp >= train_end) & (frame.timestamp < val_end)].copy(),
            frame.loc[frame.timestamp >= val_end].copy())


def split_summary(parts):
    return pd.DataFrame([{'split': name, 'rows': len(part),
                          'start': part.timestamp.min(), 'end': part.timestamp.max(),
                          'fraud_rate': part.is_fraud.mean()}
                         for name, part in zip(['train', 'validation', 'test'], parts)])
