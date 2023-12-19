"""Data partitioning script for UCF-101 dataset."""

import argparse
import json
import os
import random


# Use this script file to create data partitions
# on UCF101 dataset for federated learning from .json files.
def parse_args():
    """Parse argument to the main function."""
    parser = argparse.ArgumentParser(description="Build partition UCF101 dataset")
    parser.add_argument(
        "--json_path",
        default="/path/to/ucf101/annotations",
        type=str,
        help="directory path to training .json files.",
    )
    parser.add_argument(
        "--output_path",
        default="/path/to/ucf101/annotations/annotations_fed",
        type=str,
        help="output path for generated clients .json files.",
    )
    parser.add_argument(
        "--num_clients",
        default=10,
        type=int,
        help="number of clients for partitioning.",
    )
    parser.add_argument(
        "--seed", default=7, type=int, help="random seed for partitioning."
    )
    args = parser.parse_args()

    return args


def main():
    """Define the main function for data partitioning."""
    args = parse_args()
    json_path = args.json_path
    output_path = args.output_path
    num_clients = args.num_clients
    seed = args.seed

    random.seed(seed)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # load .json files and concatenate
    data_list = []
    # for i in range(1, 4):
    with open(json_path + "/train_split_1.json", "r") as f_r:
        json_object = json.load(f_r)
        data_list += json_object

    # data splitting to each client
    random.shuffle(data_list)
    num_data = len(data_list)
    for i in range(0, num_clients):
        client_data = data_list[
            int(i * num_data / num_clients) : int((i + 1) * num_data / num_clients)
        ]
        with open(
            os.path.join(output_path, "client_dist" + str(i + 1) + ".json"), "w"
        ) as f:
            json.dump(client_data, f, indent=2)


if __name__ == "__main__":
    main()
