import argparse

import flwr as fl
import matplotlib.pyplot as plt

from server import strategy
from client import client_fn

parser = argparse.ArgumentParser(
    description="Finetuning of a ViT with Flower Simulation."
)

parser.add_argument(
    "--num-rounds",
    type=int,
    default=20,
    help="Number of rounds.",
)


def main():
    args = parser.parse_args()

    # To control the degree of parallelism
    # With default settings in this example,
    # each client should take just ~1GB of VRAM.
    client_resources = {
        "num_cpus": 4,
        "num_gpus": 0.2,
    }

    # Launch simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=20,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

    print(history)

    # Basic plotting
    global_accuracy_centralised = history.metrics_centralized["accuracy"]
    round = [int(data[0]) for data in global_accuracy_centralised]
    acc = [100.0 * data[1] for data in global_accuracy_centralised]
    plt.plot(round, acc)
    plt.xticks(round)
    plt.grid()
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Round")
    plt.title("Federated finetuning of ViT for Flowers-102")
    plt.savefig("central_evaluation.png")


if __name__ == "__main__":
    main()
