import argparse
import pickle
import ray

from client import FlowerClient
from utils import evaluate_config
from pathlib import Path

from flwr.server import ServerConfig
from flwr.server.strategy import ResourceAwareFedAvg
from flwr.simulation import start_simulation
from flwr.common.typing import Config, Metrics, NDArrays

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_rounds", type=int, default=503)

# Flower client, adapted from Pytorch quickstart example

if __name__ == "__main__":

    args = parser.parse_args()
    pool_size = 10965

    # Get profiles
    with open(
        "/datasets/FedScale/openImg/client_data_mapping/profiles.pickle", "rb"
    ) as f:  ###TODOOO
        profiles = pickle.load(f)

    # configure the strategy
    def fit_config(server_round: int) -> Config:
        config: Config = {
            "epochs": 1,  # number of local epochs
            "batch_size": 20,
        }
        return config

    strategy = ResourceAwareFedAvg(
        fraction_fit=0.00911992704,
        fraction_evaluate=0.0,
        min_fit_clients=100,
        # min_evaluate_clients=0,
        min_available_clients=pool_size,
        on_fit_config_fn=fit_config,
        # on_evaluate_config_fn=evaluate_config,
        profiles=profiles,
        num_warmup_steps=20,
        save_models_folder=Path("/local/scratch/pedro/experiments/")
        # evaluate_fn=get_evaluate_fn(testset),  # centralized evaluation of global model
    )

    fed_dir = Path("/datasets/FedScale/openImg/")

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, str(fed_dir.absolute()))

    ray_init_args = {"include_dashboard": False}

    # start simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        use_profiler=True,
        max_workers=18,
    )
