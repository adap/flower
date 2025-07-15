"""Utils used for the FEMNIST project."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from flwr.common import Metrics, Scalar


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Dict[str, Scalar]:
    """Compute weighted average.

    It is generic implementation that averages only over floats and ints
    and drops the other data types of the Metrics.
    """
    n_batches_list = [n_batches for n_batches, _ in metrics]
    n_batches_sum = sum(n_batches_list)
    metrics_lists: Dict[str, List[float]] = {}
    for number_of_batches, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            if isinstance(value, (float, int)):
                metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for number_of_batches, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            if isinstance(value, (float, int)):
                metrics_lists[single_metric].append(float(number_of_batches * value))

    weighted_metrics: Dict[str, Scalar] = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / n_batches_sum

    return weighted_metrics


def setup_seed(seed: int):
    """Set up seed for numpy and torch.

    Parameters
    ----------
    seed: int
        random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
