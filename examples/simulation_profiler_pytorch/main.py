import argparse
import pickle
import ray

from client import FlowerClient
from dataset_utils import do_fl_partitioning, get_cifar_10
from utils import evaluate_config, fit_config

from flwr.server import ServerConfig
from flwr.server.strategy import ResourceAwareFedAvg
from flwr.simulation import start_simulation

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_cpus_per_client", type=float, default=1)  # Initial values
parser.add_argument("--num_gpus_per_client", type=float, default=0.5)  # Initial values
parser.add_argument("--num_rounds", type=int, default=5)

# Flower client, adapted from Pytorch quickstart example


# Start simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.


if __name__ == "__main__":

    args = parser.parse_args()

    pool_size = 100  # number of dataset partitions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_cpus_per_client,
        "num_gpus": args.num_gpus_per_client,
    }

    # Download CIFAR-10 dataset
    train_path, testset = get_cifar_10()

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = do_fl_partitioning(
        path_to_dataset=train_path,
        pool_size=pool_size,
        alpha=1000,
        num_classes=10,
        val_ratio=0.1,
    )

    # Get profiles
    with open(fed_dir / "profiles.pickle", "rb") as f:
        profiles = pickle.load(f)

    # configure the strategy
    strategy = ResourceAwareFedAvg(
        fraction_fit=0.1,
        fraction_evaluate=0.0,
        min_fit_clients=2,
        # min_evaluate_clients=0,
        min_available_clients=pool_size,
        on_fit_config_fn=fit_config,
        # on_evaluate_config_fn=evaluate_config,
        profiles=profiles
        # evaluate_fn=get_evaluate_fn(testset),  # centralized evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, str(fed_dir.absolute()))

    # (optional) specify Ray config
    ray.init(include_dashboard=False)

    ray_init_args = {
        "include_dashboard": False,
    }

    # start simulation
    start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        use_profiler=True,
    )
