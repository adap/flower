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
"""Flower client using TensorFlow for CIFAR-10/100."""


import argparse

import tensorflow as tf

import flower as flwr
from flower.logger import configure
from flower_benchmark.common import VisionClassificationClient, load_partition
from flower_benchmark.dataset import tf_cifar_partitioned
from flower_benchmark.model import resnet50v2

from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, SEED

tf.get_logger().setLevel("ERROR")


def main() -> None:
    """Load data, create and start CifarClient."""
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
    parser.add_argument(
        "--cifar",
        type=int,
        choices=[10, 100],
        default=10,
        help="CIFAR version, allowed values: 10 or 100 (default: 10)",
    )
    args = parser.parse_args()

    # Configure logger
    configure(f"client:{args.cid}", args.log_file, args.log_host)

    # Load model
    model = resnet50v2(input_shape=(32, 32, 3), num_classes=args.cifar, seed=SEED)

    # Load local data partition
    use_cifar100 = args.cifar == 100
    xy_partitions, xy_test = tf_cifar_partitioned.load_data(
        iid_fraction=0.0, num_partitions=args.clients, cifar100=use_cifar100
    )
    xy_train, xy_test = load_partition(
        xy_partitions,
        xy_test,
        partition=args.partition,
        num_clients=args.clients,
        seed=SEED,
        dry_run=args.dry_run,
    )

    # Start client
    client = VisionClassificationClient(
        args.cid, model, xy_train, xy_test, args.delay_factor, args.cifar
    )
    flwr.app.start_client(args.grpc_server_address, args.grpc_server_port, client)


if __name__ == "__main__":
    main()
