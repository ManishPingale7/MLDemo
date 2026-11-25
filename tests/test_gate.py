from copy import deepcopy
import pytest
from fraud.config import load_config
from fraud.gate import check_gate

GOOD = {'pr_auc': 0.4, 'recall': 0.7, 'cost_per_txn': 12.0}


def test_equal_champion_passes():
    assert check_gate(GOOD, GOOD, load_config())[0]


@pytest.mark.parametrize('key,value', [('pr_auc', 0.01), ('recall', 0.0),
                                      ('cost_per_txn', 25.0), ('cost_per_txn', 13.0),
                                      ('pr_auc', float('nan')), ('recall', float('inf'))])
def test_failed_limit_blocks(key, value):
    candidate = deepcopy(GOOD)
    candidate[key] = value
    passed, reasons = check_gate(candidate, GOOD, load_config())
    assert not passed
    assert reasons


def test_baseline_blocks():
    assert not check_gate({'pr_auc': 0.015, 'recall': 0, 'cost_per_txn': 45},
                          GOOD, load_config())[0]
