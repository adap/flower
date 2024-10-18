from authexample.task import load_data_to_disk
import argparse

if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(
        description="Load CIFAR-10 dataset partitions to disk"
    )

    # Add an optional positional argument for number of partitions
    parser.add_argument(
        "num_partitions",
        type=int,
        nargs="?",
        default=2,
        help="Number of partitions to create (default: 2)",
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided argument
    load_data_to_disk(args.num_partitions)
