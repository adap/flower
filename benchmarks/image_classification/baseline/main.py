from typing import Dict
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import yaml
import torch

import flwr as fl
from flwr.common.typing import Scalar

from datasets.utils.logging import disable_progress_bar
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from utils import apply_transforms, fit_weighted_average, set_params
from model import Net
from client import FlowerClient


def get_client_fn(dataset: FederatedDataset, test_size: float, seed: int):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Let's get the partition corresponding to the i-th client
        client_dataset = dataset.load_partition(int(cid), "train")

        # Client train/test split
        client_dataset_splits = client_dataset.train_test_split(test_size=test_size, seed=seed)
        trainset = client_dataset_splits["train"]

        # Now we apply the transform to each batch.
        trainset = trainset.with_transform(apply_transforms)

        # Create and return client
        return FlowerClient(trainset)

    return client_fn


def get_fit_config(local_epoch, batch_size, lr, momentum):
    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": local_epoch,  # Number of local epochs done by clients
            "batch_size": batch_size,  # Batch size to use by clients during fit()
            "lr": lr,  # Learning rate
            "momentum": momentum,  # Momentum value for optimiser
        }
        return config
    return fit_config


def get_evaluate_fn(total_rounds, save_path):
    """Return an evaluation function for saving global model."""
    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        # Init model
        model = Net()
        set_params(model, parameters)

        # Save model from the last round
        if server_round == total_rounds:
            torch.save(model.state_dict(), f"{save_path}/aggregated_model_{server_round}.pth")

        return 0.0, {}
    return evaluate


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    # Load static configs
    with open("../static/static_config.yaml", "r") as f:
        static_config = yaml.load(f, Loader=yaml.FullLoader)

    # Download CIFAR10 dataset and partition it
    partitioner = IidPartitioner(num_partitions=static_config["num_clients"])
    cifar10_fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})

    # Configure the strategy
    strategy = instantiate(
        cfg.strategy,
        fraction_fit=static_config["fraction_fit"],  # Sample 10% of available clients for training
        fraction_evaluate=0.0,  # no evaluation
        min_fit_clients=static_config["min_fit_clients"],  # Never sample less than 10 clients for training
        on_fit_config_fn=get_fit_config(static_config["local_epoch"], cfg.batch_size, cfg.lr, cfg.momentum),
        fit_metrics_aggregation_fn=fit_weighted_average,  # Weighted average fit metrics
        evaluate_fn=get_evaluate_fn(static_config["num_rounds"], save_path),  # Evaluation function to save global model
    )

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": cfg.client_resources.num_cpus,
        "num_gpus": cfg.client_resources.num_gpus,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(cifar10_fds, static_config["test_size"], static_config["seed"]),
        num_clients=static_config["num_clients"],
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=static_config["num_rounds"]),
        strategy=strategy,
        actor_kwargs={
            "on_actor_init_fn": disable_progress_bar  # disable tqdm on each actor/process spawning virtual clients
        },
    )


if __name__ == "__main__":
    main()
