import argparse


import flwr as fl

from client import FlowerClient
from server import FedAvgWithWandB, get_evaluate_fn, weighted_average
from utils import prepare_dataset

parser = argparse.ArgumentParser(description="Flower + W&B + PyTorch")

parser.add_argument("--num_rounds", type=int, default=20, help="Number of FL rounds (default = 20)")
parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to assign to a virtual client")
parser.add_argument("--num_gpus", type=float, default=0.0, help="Ratio of GPU memory to assign to a virtual client")

TOTAL_CLIENTS = 100


def get_client_fn(train_partitions, val_partitions):
    """Return a function to be executed by the VirtualClientEngine in order to construct
    a client."""

    def client_fn(cid: str):
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with id = cid
        trainset, valset = train_partitions[int(cid)], val_partitions[int(cid)]

        # Create and return client
        return FlowerClient(cid, trainset, valset)

    return client_fn


def main():
    # Parse input arguments
    args = parser.parse_args()

    clients_trainset, clients_valset, testset = prepare_dataset(TOTAL_CLIENTS)

    # Resources to be assigned to each virtual client
    client_resources = {
        "num_cpus": args.num_cpus,
        "num_gpus": args.num_gpus,
    }

    # A strategy that behaves mostly like FedAvg but logs
    # evaluate results to Weight and Biases
    strategy = FedAvgWithWandB(
        fraction_fit=0.1,  # Sample 10% of available clients for training
        fraction_evaluate=0.1,  # Sample 5% of available clients for evaluation
        min_fit_clients=10,  # Never sample less than 10 clients for training
        min_evaluate_clients=10,  # Never sample less than 10 clients for evaluation
        min_available_clients=TOTAL_CLIENTS,
        evaluate_fn=get_evaluate_fn(testset),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(clients_trainset, clients_valset),
        num_clients=TOTAL_CLIENTS,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
