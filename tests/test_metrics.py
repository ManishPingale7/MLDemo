import numpy as np
import pytest
from fraud.config import load_config
from fraud.evaluate import best_threshold, metrics, threshold_table


def test_known_counts():
    result = metrics([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9], 0.5, load_config())
    assert result['precision'] == result['recall'] == result['accuracy'] == 0.5
    assert result['cost_per_txn'] == 775
    assert result['pr_auc'] == pytest.approx(5 / 6)
    assert result['roc_auc'] == 0.75


def test_baseline():
    result = metrics([0, 0, 0, 1], np.zeros(4), 0.5, load_config())
    assert result['recall'] == result['precision'] == 0
    assert result['pr_auc'] == 0.25


def test_daily_budget_not_just_average():
    config = load_config()
    config['alert_budget_per_day'] = 1
    y, p, dates = [1, 1, 0, 0], [0.9, 0.8, 0.1, 0.1], [1, 1, 2, 3]
    table = threshold_table(y, p, config, dates)
    assert not table.loc[table.threshold < 0.8, 'within_budget'].any()
    threshold, _ = best_threshold(y, p, config, dates)
    assert threshold > 0.8
