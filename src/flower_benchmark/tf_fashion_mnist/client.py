# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Flower client using TensorFlow for Fashion-MNIST image classification."""


import argparse

import tensorflow as tf

import flower as flwr
from flower.logger import configure
from flower_benchmark.common import VisionClassificationClient, load_partition
from flower_benchmark.dataset import tf_fashion_mnist_partitioned
from flower_benchmark.model import orig_cnn

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, SEED

tf.get_logger().setLevel("ERROR")


def main() -> None:
    """Load data, create and start FashionMnistClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--grpc_server_address",
        type=str,
        default=DEFAULT_GRPC_SERVER_ADDRESS,
        help="gRPC server address (IPv6, default: [::])",
    )
    parser.add_argument(
        "--grpc_server_port",
        type=int,
        default=DEFAULT_GRPC_SERVER_PORT,
        help="gRPC server port (default: 8080)",
    )
    parser.add_argument(
        "--cid", type=str, required=True, help="Client CID (no default)"
    )
    parser.add_argument(
        "--partition", type=int, required=True, help="Partition index (no default)"
    )
    parser.add_argument(
        "--clients", type=int, required=True, help="Number of clients (no default)",
    )
    parser.add_argument(
        "--delay_factor",
        type=float,
        default=0.0,
        help="Delay factor increases the time batches take to compute (default: 0.0)",
    )
    parser.add_argument(
        "--dry_run", type=bool, default=False, help="Dry run (default: False)"
    )
    parser.add_argument(
        "--log_file", type=str, help="Log file path (no default)",
    )
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    args = parser.parse_args()

    # Configure logger
    configure(f"client:{args.cid}", args.log_file, args.log_host)

    # Load model
    model = orig_cnn(input_shape=(28, 28, 1), seed=SEED)

    # Load local data partition
    xy_partitions, xy_test = tf_fashion_mnist_partitioned.load_data(
        iid_fraction=0.0, num_partitions=args.clients
    )
    xy_train, xy_test = load_partition(
        xy_partitions,
        xy_test,
        partition=args.partition,
        num_clients=args.clients,
        dry_run=args.dry_run,
        seed=SEED,
    )

    # Start client
    client = VisionClassificationClient(
        args.cid, model, xy_train, xy_test, args.delay_factor, 10
    )
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


if __name__ == "__main__":
    main()
