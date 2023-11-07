from typing import Dict

import flwr as fl
from strategy import XGbBagging

pool_size = 2
num_rounds = 5
num_clients_per_round = 2
min_evaluate_clients = 2


def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


# Define strategy
strategy = XGbBagging(
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    fraction_evaluate=1.0,
    min_evaluate_clients=min_evaluate_clients,
    on_evaluate_config_fn=eval_config,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
