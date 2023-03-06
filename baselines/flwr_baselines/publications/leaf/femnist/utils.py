from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    n_batches_list = [n_batches for n_batches, _ in metrics]
    n_batches_sum = sum(n_batches_list)
    metrics_lists: Dict[str, List[Union[float, int]]] = {}
    for number_of_batches, all_metrics_dict in metrics:
        #  Calculate each metric one by one
        for single_metric, _ in all_metrics_dict.items():
            metrics_lists[single_metric] = []
        # Just one iteration needed to initialize the keywords
        break

    for number_of_batches, all_metrics_dict in metrics:
        # Calculate each metric one by one
        for single_metric, value in all_metrics_dict.items():
            # Add weighted metric
            metrics_lists[single_metric].append(number_of_batches * value)

    weighted_metrics = {}
    for metric_name, metric_values in metrics_lists.items():
        weighted_metrics[metric_name] = sum(metric_values) / n_batches_sum

    return weighted_metrics


def setup_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
