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
"""Flower Datasets command line interface `create` command."""


from pathlib import Path
from typing import Annotated

import click
import typer

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner


def create(
    name: Annotated[
        str,
        typer.Option(
            "--name",
            help="Hugging Face dataset identifier (e.g., 'ylecun/mnist').",
        ),
    ],
    num_partitions: Annotated[
        int,
        typer.Option(
            "--num-partitions",
            min=1,
            help="Number of partitions to create.",
        ),
    ] = 10,
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out-dir",
            help="Output directory for the federated dataset.",
        ),
    ] = Path("./federated_dataset"),
) -> None:
    """Create a federated dataset and save it to disk.

    This command is used to generate federated datasets
    for demo purposes and currently supports only IID
    partitioning `IidPartitioner`.
    """
    # Validate number of partitions
    if num_partitions <= 0:
        raise click.ClickException("--num-partitions must be a positive integer.")

    # Handle output directory
    if out_dir.exists():
        overwrite = typer.confirm(
            typer.style(
                f"\n💬 {out_dir} already exists, do you want to override it?",
                fg=typer.colors.MAGENTA,
                bold=True,
            ),
            default=False,
        )
        if not overwrite:
            return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Create data partitioner
    partitioner = IidPartitioner(num_partitions=num_partitions)

    # Create the federated dataset
    fds = FederatedDataset(dataset=name, partitioners={"train": partitioner})

    # Load partitions and save them to disk
    for partition_id in range(num_partitions):
        partition = fds.load_partition(partition_id=partition_id)
        partition_out_dir = out_dir / f"partition_{partition_id}"
        partition.save_to_disk(partition_out_dir)
