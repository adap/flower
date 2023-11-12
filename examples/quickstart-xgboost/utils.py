import argparse


def client_args_parser():
    """Parse arguments to define experimental settings on client side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_partitions", default=20, type=int, help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner_type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--partition_id",
        default=0,
        type=int,
        help="Partition ID used for the current client.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test_fraction",
        default=0.2,
        type=float,
        help="Test fraction for train/test splitting.",
    )
    parser.add_argument(
        "--centralised_eval",
        action='store_true',
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args


def server_args_parser():
    """Parse arguments to define experimental settings on server side."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pool_size", default=2, type=int, help="Number of total clients."
    )
    parser.add_argument(
        "--num_rounds", default=5, type=int, help="Number of FL rounds."
    )
    parser.add_argument(
        "--num_clients_per_round",
        default=2,
        type=int,
        help="Number of clients participate in training each round.",
    )
    parser.add_argument(
        "--num_evaluate_clients",
        default=2,
        type=int,
        help="Number of clients selected for evaluation.",
    )
    parser.add_argument(
        "--centralised_eval",
        action='store_true',
        help="Conduct centralised evaluation (True), or client evaluation on hold-out data (False).",
    )

    args = parser.parse_args()
    return args
