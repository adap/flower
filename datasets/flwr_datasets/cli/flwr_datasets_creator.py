# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""`flwr-datasets-creator` command."""


import argparse
import shutil
from pathlib import Path

import typer
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def flwr_datasets_creator() -> None:
    """Create a federated dataset and save it to disk.

    This command is used to generated demo data, and currently supports
    only IID partitioning via `IidPartitioner`.
    """
    parser = _parse_args_run_creator()
    args = parser.parse_args()

    # Validate number of partitions
    if args.num_partitions <= 0:
        parser.error("--num-partitions must be a positive integer.")

    # Handle output directory
    if args.out_dir.exists():
        overwrite = typer.confirm(
            f"Output directory '{args.out_dir}' already exists. Overwrite?",
            default=False,
        )
        if not overwrite:
            typer.echo("Aborting.")
            return

        shutil.rmtree(args.out_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Create data partitioner
    partitioner = IidPartitioner(num_partitions=args.num_partitions)

    # Create the federated dataset
    fds = FederatedDataset(
        dataset=args.name,
        partitioners={"train": partitioner},
    )

    # Load partitions and save them to disk
    for partition_id in range(args.num_partitions):
        partition = fds.load_partition(partition_id=partition_id)
        out_dir = args.out_dir / f"partition_{partition_id}"
        partition.save_to_disk(out_dir)


def _parse_args_run_creator() -> argparse.ArgumentParser:
    """Parse flwr-datasets-creator command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create federated dataset partitions and save them to disk.",
    )
    parser.add_argument(
        "--name",
        required=True,
        type=str,
        help="Hugging Face dataset identifier (e.g., 'username/dataset_name').",
    )
    parser.add_argument(
        "--num-partitions",
        default=10,
        type=int,
        help="Number of partitions to create for the federated dataset.",
    )
    parser.add_argument(
        "--out-dir",
        default=Path("./federated_dataset"),
        type=Path,
        help="Output directory for the federated dataset.",
    )

    return parser
