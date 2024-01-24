from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
import torch

import utils

import warnings

from flwr_datasets import FederatedDataset

warnings.filterwarnings("ignore")


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data here to avoid the overhead of doing it in `evaluate` itself
    centralized_data = utils.load_centralized_data()
    if toy:
        # use only 10 samples as validation set
        centralized_data = centralized_data.select(range(10))

    val_loader = DataLoader(centralized_data, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, val_loader)
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        action="store_true",
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "alexnet"],
        help="Use either Efficientnet or Alexnet models. \
             If you want to achieve differential privacy, please use the Alexnet model",
    )

    args = parser.parse_args()

    if args.model == "alexnet":
        model = utils.load_alexnet(classes=10)
    else:
        model = utils.load_efficientnet(classes=10)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=10,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
