from copy import deepcopy
import pytest
from fraud.config import load_config
from fraud.gate import check_gate

GOOD = {'pr_auc': 0.94, 'precision': 0.94, 'recall': 0.94, 'cost_per_txn': 3.0}


def test_equal_champion_passes():
    assert check_gate(GOOD, GOOD, load_config())[0]


@pytest.mark.parametrize('key,value', [('pr_auc', 0.01), ('recall', 0.0),
                                      ('cost_per_txn', 25.0), ('cost_per_txn', 4.0), ('precision', 0.80),
                                      ('precision', float('nan')),
                                      ('pr_auc', float('nan')), ('recall', float('inf'))])
def test_failed_limit_blocks(key, value):
    candidate = deepcopy(GOOD)
    candidate[key] = value
    passed, reasons = check_gate(candidate, GOOD, load_config())
    assert not passed
    assert reasons


def test_baseline_blocks():
    assert not check_gate({'pr_auc': 0.015, 'precision': 0, 'recall': 0, 'cost_per_txn': 45},
                          GOOD, load_config())[0]


def test_model_selection_rejects_cheaper_low_precision():
    import pandas as pd
    from fraud.models import select_model
    low_precision = {**GOOD, 'precision': 0.40, 'cost_per_txn': 0.0}
    table = pd.DataFrame([GOOD, low_precision, GOOD],
                         index=['Always approve', 'Cheaper but noisy', 'Qualified'])
    bundles = {name: {'name': name} for name in table.index}
    assert select_model(bundles, table, load_config())['name'] == 'Qualified'


def test_model_selection_requires_a_qualified_model():
    import pandas as pd
    from fraud.models import select_model
    table = pd.DataFrame([{**GOOD, 'precision': 0.1}, {**GOOD, 'recall': 0.1}],
                         index=['Always approve', 'Weak'])
    with pytest.raises(ValueError, match='No model meets'):
        select_model({}, table, load_config())
