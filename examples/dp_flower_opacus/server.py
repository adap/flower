"""Script to start a server from the command line.

Author: Alex Kyllo
"""
from argparse import ArgumentParser, Namespace
from functools import partial

from main import Net, evaluate, get_weights, load_data, start_server
from torch.nn import CrossEntropyLoss

import flwr as fl


def get_server_args() -> Namespace:
    """Get command line args for the server."""
    parser = ArgumentParser(description="Flower Server for DP demo.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of rounds for the federated training"
    )
    parser.add_argument(
        "--min_fit_clients",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )
    parser.add_argument(
        "--available_clients",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_server_args()
    batch_size = int(args.batch_size)
    min_fit_clients = int(args.min_fit_clients)
    available_clients = int(args.available_clients)
    rounds = int(args.rounds)
    net = Net()
    init_weights = get_weights(net)
    # Convert the weights (np.ndarray) to parameters
    init_param = fl.common.weights_to_parameters(init_weights)
    _, test_loader = load_data(batch_size)
    eval_fn = partial(evaluate, net, CrossEntropyLoss(), test_loader, "cpu")
    server_process = start_server(init_param, min_fit_clients, available_clients, rounds, eval_fn)
