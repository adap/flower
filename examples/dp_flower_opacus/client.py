import multiprocessing as mp
from argparse import ArgumentParser, Namespace
from functools import partial

from main import start_client


def get_args() -> Namespace:
    """Get command line args for the client."""
    parser = ArgumentParser(description="Flower Client for DP demo.")
    parser.add_argument(
        "cid",
        type=int,
        help="The current client ID. Used for splitting the dataset.",
    )
    parser.add_argument(
        "--num-clients",
        default=2,
        type=int,
        help="Total number of clients for federated training.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of local epochs to train per round.",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.01, type=float, help="Learning rate for training"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Target epsilon for the privacy budget.",
    )
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Gradient clipping norm")
    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of rounds for the federated training."
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Which client device to run training on."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # get the command line arg values
    args = get_args()
    num_clients = int(args.num_clients)
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    rounds = int(args.rounds)
    target_epsilon = float(args.eps)
    max_grad_norm = float(args.max_grad_norm)
    learning_rate = float(args.learning_rate)
    cid = int(args.cid)
    # start the client
    start_client(
        batch_size,
        epochs,
        rounds,
        num_clients,
        "cpu",
        target_epsilon,
        max_grad_norm,
        learning_rate,
        cid,
    )
