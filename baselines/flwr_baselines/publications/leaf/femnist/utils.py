from typing import List, Tuple

from src.py.flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    n_batches_list = [n_batches for n_batches, _ in metrics]
    n_batches_sum = sum(n_batches_list)
    metrics_lists = {}
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
