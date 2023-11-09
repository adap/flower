import argparse


def client_args_parser():
    """Parse arguments to define experimental settings."""
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
        help="test fraction for train/test splitting.",
    )

    args = parser.parse_args()
    return args
