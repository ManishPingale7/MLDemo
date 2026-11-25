"""One synthetic table; all distribution choices are teaching assumptions."""
from collections import deque

import numpy as np
import pandas as pd

CATEGORIES = ['grocery', 'food', 'fuel', 'travel', 'shopping', 'utilities',
              'electronics', 'gaming']
CITIES = ['Mumbai', 'Delhi', 'Bengaluru', 'Chennai', 'Kolkata', 'Hyderabad',
          'Pune', 'Jaipur', 'Ahmedabad', 'Lucknow']
COORDS = np.array([[19.08, 72.88], [28.61, 77.21], [12.97, 77.59],
                   [13.08, 80.27], [22.57, 88.36], [17.39, 78.49],
                   [18.52, 73.86], [26.91, 75.79], [23.02, 72.57], [26.85, 80.95]])
COLUMNS = ['txn_id', 'timestamp', 'customer_id', 'amount_inr', 'merchant_category',
           'city', 'distance_from_home_km', 'device_type', 'is_new_merchant',
           'txns_last_1h', 'customer_avg_amount_inr', 'is_fraud']


def previous_hour_counts(customers, seconds):
    """Count strictly earlier transactions; the current row is never counted."""
    queues = {}
    counts = np.zeros(len(customers), dtype=int)
    for i, (customer, second) in enumerate(zip(customers, seconds)):
        queue = queues.setdefault(customer, deque())
        while queue and queue[0] < second - 3600:
            queue.popleft()
        counts[i] = len(queue)
        queue.append(second)
    return counts


def generate_data(config, preset='full'):
    """Generate routine/risky profiles, lightly noisy labels, then mild defects."""
    rng = np.random.default_rng(config['seed'])
    n = config['presets'][preset]['rows']
    customers = config['presets'][preset]['customers']
    start = pd.Timestamp(config['data']['start'])
    typical = rng.lognormal(np.log(500), 0.55, customers)
    home = rng.integers(0, len(CITIES), customers)
    preferred = rng.integers(0, len(CATEGORIES), customers)
    customer = rng.integers(0, customers, n)
    # A rare, coherent risk profile makes this an intentionally easy teaching dataset.
    risky_probability = config['data']['risky_fraud_probability']
    routine_probability = config['data']['routine_fraud_probability']
    profile_rate = ((config['data']['fraud_rate'] - routine_probability) /
                    (risky_probability - routine_probability))
    risky_profile = rng.random(n) < profile_rate

    # Evening, weekend and month-start activity are explicit assumptions.
    dates = pd.date_range(start, periods=config['data']['days'])
    day_weight = 1 + 0.25 * (dates.dayofweek >= 5) + 0.2 * (dates.day <= 3)
    day = rng.choice(len(dates), n, p=day_weight / day_weight.sum())
    hour_weight = np.ones(24)
    hour_weight[:6] = 0.35
    hour_weight[18:23] = 2.2
    hour = rng.choice(24, n, p=hour_weight / hour_weight.sum())
    late = risky_profile & (rng.random(n) < 0.75)
    hour[late] = rng.integers(0, 6, late.sum())
    seconds = day * 86400 + hour * 3600 + rng.integers(0, 3600, n)
    # Small shopping bursts make actual previous-hour counts informative.
    for i in range(1, n):
        if rng.random() < (0.55 if risky_profile[i] else 0.18):
            customer[i] = customer[i - 1]
            seconds[i] = min(seconds[i - 1] + rng.integers(1, 900), 90 * 86400 - 1)
    order = np.argsort(seconds, kind='stable')
    seconds, customer = seconds[order], customer[order]
    risky_profile = risky_profile[order]
    timestamp = start + pd.to_timedelta(seconds, unit='s')
    counts = previous_hour_counts(customer, seconds)
    category = np.where(rng.random(n) < 0.65, preferred[customer],
                        rng.integers(0, len(CATEGORIES), n))
    city = np.where(rng.random(n) < 0.90, home[customer], rng.integers(0, 10, n))
    # Risky profiles combine large relative amounts with a genuinely different city.
    category[risky_profile] = rng.choice([4, 6, 7], risky_profile.sum(), p=[0.15, 0.50, 0.35])
    city[risky_profile] = (home[customer[risky_profile]] +
                          rng.integers(1, len(CITIES), risky_profile.sum())) % len(CITIES)
    # Rough geographic distance; coordinates and this approximation are assumptions.
    delta = (COORDS[city] - COORDS[home[customer]]) * [111, 102]
    distance = np.sqrt((delta ** 2).sum(axis=1)) + rng.exponential(5, n)
    amount = typical[customer] * rng.lognormal(-0.25, 0.75, n)
    tail = rng.random(n) < 0.04
    amount[tail] *= rng.uniform(4, 15, tail.sum())
    rounded = rng.random(n) < 0.12
    amount[rounded] = rng.choice([100, 500, 1000], rounded.sum())
    amount[risky_profile] = typical[customer[risky_profile]] * rng.uniform(7, 15, risky_profile.sum())
    amount = amount.round(2)
    device = rng.choice(['mobile', 'desktop', 'pos_terminal'], n, p=[0.60, 0.15, 0.25])
    device[risky_profile] = rng.choice(['desktop', 'mobile'], risky_profile.sum(), p=[0.85, 0.15])
    # Internal merchant identifiers exist only during generation, not as a second table.
    merchant = np.where(rng.random(n) < 0.85, category, category * 100 +
                        rng.integers(10, 80, n))
    merchant[risky_profile] = n + np.flatnonzero(risky_profile)
    new = (~pd.DataFrame({'c': customer, 'm': merchant}).duplicated()).astype(int).to_numpy()
    # Profile membership is NOT an output column or a model input.
    # Light label noise prevents a perfect mapping from profile to fraud.
    probability = np.where(risky_profile, risky_probability, routine_probability)
    fraud = rng.binomial(1, probability)
    frame = pd.DataFrame({
        'txn_id': [f'T{i:06d}' for i in range(n)], 'timestamp': timestamp,
        'customer_id': customer, 'amount_inr': amount,
        'merchant_category': np.array(CATEGORIES)[category], 'city': np.array(CITIES)[city],
        'distance_from_home_km': distance.round(2), 'device_type': device,
        'is_new_merchant': new, 'txns_last_1h': counts,
        'customer_avg_amount_inr': typical[customer].round(2), 'is_fraud': fraud,
    })
    frame.loc[rng.choice(n, round(n * 0.025), replace=False), 'distance_from_home_km'] = np.nan
    frame.loc[rng.choice(n, round(n * 0.02), replace=False), 'device_type'] = None
    messy = rng.choice(n, round(n * 0.01), replace=False)
    frame.loc[messy, 'merchant_category'] = ' ' + frame.loc[messy, 'merchant_category'].str.upper() + ' '
    invalid = rng.choice(n, max(4, round(n * 0.0002)), replace=False)
    frame.loc[invalid, 'amount_inr'] = rng.choice([-100.0, 0.0], len(invalid))
    duplicates = frame.iloc[rng.choice(n, round(n * 0.003), replace=False)]
    return pd.concat([frame, duplicates], ignore_index=True)[COLUMNS]
