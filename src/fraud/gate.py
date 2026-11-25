"""A readable candidate acceptance rule; never evaluates the final test set."""
import math
from fraud.evaluate import metrics
from fraud.features import FEATURES


def check_gate(candidate, champion, config):
    limits = config['gate']
    reasons = []
    for key in ['pr_auc', 'precision', 'recall', 'cost_per_txn']:
        if not math.isfinite(candidate[key]):
            reasons.append(f'{key} is not finite')
    if candidate['pr_auc'] < limits['min_pr_auc']:
        reasons.append(f'PR-AUC {candidate["pr_auc"]:.3f} < {limits["min_pr_auc"]:.3f}')
    if candidate['precision'] < limits['min_precision']:
        reasons.append(f'precision {candidate["precision"]:.3f} < {limits["min_precision"]:.3f}')
    if candidate['recall'] < limits['min_recall']:
        reasons.append(f'recall {candidate["recall"]:.3f} < {limits["min_recall"]:.3f}')
    if candidate['cost_per_txn'] > limits['max_cost']:
        reasons.append(f'cost INR {candidate["cost_per_txn"]:.2f} > {limits["max_cost"]:.2f}')
    if not math.isfinite(champion['cost_per_txn']):
        reasons.append('champion cost is not finite')
    elif candidate['cost_per_txn'] > champion['cost_per_txn'] + 1e-9:
        reasons.append(f'cost INR {candidate["cost_per_txn"]:.2f} exceeds champion '
                       f'INR {champion["cost_per_txn"]:.2f}')
    return not reasons, reasons or ['All limits passed; candidate cost is no higher than champion.']


def evaluate_bundle(bundle, validation, config):
    probability = bundle['pipeline'].predict_proba(validation[FEATURES])[:, 1]
    return metrics(validation.is_fraud, probability, bundle['threshold'], config)
