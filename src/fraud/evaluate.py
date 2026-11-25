"""Shared metric definitions for notebooks, tracking and quality gates."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)


def metrics(y, probability, threshold, config):
    predicted = np.asarray(probability) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    cost = (fn * config['costs']['missed_fraud'] + fp * config['costs']['false_alarm']) / len(y)
    return {'accuracy': accuracy_score(y, predicted),
            'precision': precision_score(y, predicted, zero_division=0),
            'recall': recall_score(y, predicted, zero_division=0),
            'f1': f1_score(y, predicted, zero_division=0),
            'pr_auc': average_precision_score(y, probability),
            'roc_auc': roc_auc_score(y, probability), 'cost_per_txn': float(cost),
            'alerts': int(tp + fp)}


def threshold_table(y, probability, config, days=15, scale=1.0):
    """Vectorized fixed grid: validation only, with a worst-day alert cap."""
    settings = config['thresholds']
    thresholds = np.linspace(settings['min'], settings['max'], settings['count'])
    y = np.asarray(y).astype(bool)
    predicted = np.asarray(probability)[:, None] >= thresholds
    tp = (predicted & y[:, None]).sum(axis=0)
    fp = (predicted & ~y[:, None]).sum(axis=0)
    fn = ((~predicted) & y[:, None]).sum(axis=0)
    # Caller supplies actual calendar dates to enforce the budget on EVERY day.
    if np.isscalar(days):
        peak = predicted.sum(axis=0) / days
    else:
        dates = np.asarray(days)
        peak = np.vstack([predicted[dates == date].sum(axis=0)
                          for date in np.unique(dates)]).max(axis=0)
    cost = (fn * config['costs']['missed_fraud'] + fp * config['costs']['false_alarm']) / len(y)
    table = pd.DataFrame({'threshold': thresholds,
                          'precision': tp / np.maximum(tp + fp, 1),
                          'recall': tp / max(y.sum(), 1), 'cost_per_txn': cost,
                          'peak_alerts_per_day': peak})
    table['within_budget'] = peak <= config['alert_budget_per_day'] * scale
    return table


def best_threshold(y, probability, config, dates, scale=1.0):
    table = threshold_table(y, probability, config, dates, scale)
    eligible = table.loc[table.within_budget]
    if eligible.empty:
        raise ValueError('No threshold satisfies the daily alert budget.')
    high_precision = eligible.loc[eligible.precision >= config['thresholds']['min_precision']]
    # Still report an unqualified model if no threshold reaches the precision target.
    # select_model and the gate will reject it.
    if not high_precision.empty:
        eligible = high_precision
    # Ties use the lower threshold, retaining more fraud for the same cost.
    best = eligible.sort_values(['cost_per_txn', 'threshold']).iloc[0]
    return float(best.threshold), table
