import torch
from datasets import Dataset
from torch.utils.data import DataLoader
import flwr as fl

from dataset import apply_eval_transforms, get_dataset_with_partitions
from model import get_model, set_parameters, test


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "lr": 0.01,  # Learning rate used by clients
        "batch_size": 32,  # Batch size to use by clients during fit()
    }
    return config


def get_evaluate_fn(
    centralized_testset: Dataset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(server_round, parameters, config):
        """Use the entire Oxford Flowers-102 test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_model()
        set_parameters(model, parameters)
        model.to(device)

        # Apply transform to dataset
        testset = centralized_testset.with_transform(apply_eval_transforms)

        testloader = DataLoader(testset, batch_size=128)
        # Run evaluation
        loss, accuracy = test(model, testloader, device=device)

        return loss, {"accuracy": accuracy}

    return evaluate


# Downloads and partition dataset
_, centralized_testset = get_dataset_with_partitions(num_partitions=20)

# Configure the strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,  # Sample 50% of available clients for training each round
    fraction_evaluate=0.0,  # No federated evaluation
    on_fit_config_fn=fit_config,
    evaluate_fn=get_evaluate_fn(centralized_testset),  # Global evaluation function
)

# To be used with Flower Next
app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
