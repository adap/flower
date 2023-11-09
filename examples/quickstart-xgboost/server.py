from typing import Dict
from logging import INFO
import xgboost as xgb

import flwr as fl
from flwr.common.logger import log
from flwr.common import Parameters, Scalar
from flwr_datasets import FederatedDataset

from strategy import XGbBagging
from utils import server_args_parser
from dataset import resplit, transform_dataset_to_dmatrix


# Parse arguments for experimental settings
args = server_args_parser()
pool_size = args.pool_size
num_rounds = args.num_rounds
num_clients_per_round = args.num_clients_per_round
num_evaluate_clients = args.num_evaluate_clients
centralised_eval = args.centralised_eval

# Load centralised test set
fds = FederatedDataset(
    dataset="jxie/higgs", partitioners={"train": 20}, resplitter=resplit
)
test_set = fds.load_full("test")
test_set.set_format("numpy")
test_dmatrix = transform_dataset_to_dmatrix(test_set)

# Hyper-parameters used for initialisation
params = {
    "objective": "binary:logistic",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


def eval_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with global epochs."""
    config = {
        "global_round": str(rnd),
    }
    return config


def evaluate_metrics_aggregation(eval_metrics):
    """Return an aggregated metric (AUC) for evaluation."""
    total_num = sum([num for num, _ in eval_metrics])
    auc_aggregated = (
        sum([metrics["AUC"] * num for num, metrics in eval_metrics]) / total_num
    )
    metrics_aggregated = {"AUC": auc_aggregated}
    return metrics_aggregated


def get_evaluate_fn(test_data):
    def evaluate_fn(
        server_round: int, parameters: Parameters, config: Dict[str, Scalar]
    ):
        # If at the first round, skip the evaluation
        if server_round == 0:
            return 0, {}
        else:
            bst = xgb.Booster(params=params)
            for para in parameters.tensors:
                para_b = bytearray(para)

            # Load global model
            bst.load_model(para_b)
            # Run evaluation
            eval_results = bst.eval_set(
                evals=[(test_data, "valid")],
                iteration=bst.num_boosted_rounds() - 1,
            )
            auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)
            log(INFO, f"AUC = {auc} at round {server_round}")

            return 0, {"AUC": auc}

    return evaluate_fn


# Define strategy
strategy = XGbBagging(
    evaluate_function=get_evaluate_fn(test_dmatrix) if centralised_eval else None,
    fraction_fit=(float(num_clients_per_round) / pool_size),
    min_fit_clients=num_clients_per_round,
    min_available_clients=pool_size,
    min_evaluate_clients=num_evaluate_clients if not centralised_eval else 0,
    fraction_evaluate=1.0 if not centralised_eval else 0.0,
    on_evaluate_config_fn=eval_config,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation
    if not centralised_eval
    else None,
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=num_rounds),
    strategy=strategy,
)
