"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""
from typing import List, Tuple, Dict

from flwr.common.typing import Metrics


def _update_dict(dictionary, key, value):

    if key in dictionary:
        dictionary[key] += value
    else:
        dictionary[key] = value


# Define metric aggregation function
def get_metrics_aggregation_fn():
    """Return function to compute metrics average.
    
    It is used for both fit() metrics and evaluate() metrics."""
    def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        """Compute per-dataset average accuracy and loss"""

        # compute per-dataset accuracy and loss
        totals: Dict[str, int] = {}
        accuracies: Dict[str, float] = {}
        losses: Dict[str, float] = {}
        pre_train_accuracies: Dict[str, float] = {}
        pre_train_losses: Dict[str, float] = {}

        for num_examples, m in metrics:
            dataset_name = m['dataset_name']
            
            _update_dict(totals, dataset_name, num_examples)
            _update_dict(accuracies, dataset_name, num_examples * m['accuracy'])
            _update_dict(losses, dataset_name, num_examples * m['loss'])
            if "pre_train_acc" in m:
                _update_dict(pre_train_accuracies, dataset_name, num_examples * m['pre_train_acc'])
            if "pre_train_loss" in m:
                _update_dict(pre_train_losses, dataset_name, num_examples * m['pre_train_loss'])

        # now normalise by the amount of total data used in the round
        total_examples = sum(totals.values())
        accuracies.update((k, v / total_examples) for k,v in accuracies.items())
        losses.update((k, v / total_examples) for k,v in losses.items())

        to_return = {"accuracy": accuracies, "losses": losses}

        if "pre_train_acc" in m:
            pre_train_accuracies.update((k, v / total_examples) for k,v in pre_train_accuracies.items())
            to_return = {**to_return, "pre_train_accuracies": pre_train_accuracies}
        if "pre_train_loss" in m:
            pre_train_losses.update((k, v / total_examples) for k,v in pre_train_losses.items())
            to_return = {**to_return, "pre_train_losses": pre_train_losses}

        return to_return

    return weighted_average

def get_on_fit_config():
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = {}
        fit_config["round"] = server_round  # add round info
        return fit_config

    return fit_config_fn
