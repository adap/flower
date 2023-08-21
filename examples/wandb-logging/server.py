import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import wandb
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Metrics, NDArrays, Parameters, Scalar

from utils import Net, test, prepare_dataset

parser = argparse.ArgumentParser(description="Flower + W&B + PyTorch")

parser.add_argument("--num_rounds", type=int, default=20, help="Number of FL rounds (default = 20)")

PROJECT_NAME = "Flower_WandB_logging_example"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def get_evaluate_fn(testset,):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters:NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire CIFAR-10 test set for evaluation."""

        model = Net()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=DEVICE)

        # return statistics
        return loss, {"entralised_evaluate_accuracy": accuracy}

    return evaluate


class FedAvgWithWandB(FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # init W&B
        now = datetime.now().strftime('%b%d_%H_%M_%S')
        wandb_project = PROJECT_NAME
        wandb_group = f'exp_{now}'
        wandb.init(project=PROJECT_NAME, name='server', group=f'exp_{now}')

        # minimal way of preparing config to send to client
        self.on_fit_config_fn = lambda round : {'project': wandb_project,
                                                'group':  wandb_group,
                                                'round': round
                                                } 

    
    def evaluate(self, server_round: int, parameters: Parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        # Log to W&B centralised loss and metrics
        wandb.log({'centralised_evaluate_loss': loss, **metrics}, step=server_round)
        return loss, metrics
    
    def aggregate_evaluate(self, server_round: int, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Log to W&B federated loss and metrics
        wandb.log({'federated_evaluate_loss': loss, **metrics}, step=server_round)
        return loss, metrics


def main():
    # Parse input arguments
    args = parser.parse_args()

    testset = prepare_dataset()

    # A strategy that behaves mostly like FedAvg but logs
    # evaluate results to Weight and Biases
    strategy = FedAvgWithWandB(evaluate_fn=get_evaluate_fn(testset),
                               evaluate_metrics_aggregation_fn=weighted_average)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
