# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""VerticalSizePartitioner class."""
# flake8: noqa: E501
# pylint: disable=C0301, R0902, R0913
from math import floor
from typing import Literal, Optional, Union, cast

import numpy as np

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner
from flwr_datasets.partitioner.vertical_partitioner_utils import (
    _add_active_party_columns,
)


class VerticalSizePartitioner(Partitioner):
    """Creates vertical partitions by spliting features (columns) based on sizes.

    The sizes refer to the number of columns after the `drop_columns` are
    dropped. `shared_columns` and `active_party_column` are excluded and
    added only after the size-based division.

    Enables selection of "active party" column(s) and palcement into
    a specific partition or creation of a new partition just for it.
    Also enables droping columns and sharing specified columns across
    all partitions.

    Parameters
    ----------
    partition_sizes : Union[list[int], list[float]]
        A list where each value represents the size of a partition.
        list[int] -> each value represent an absolute number of columns. Size zero is
        allowed and will result in an empty partition if no shared columns are present.
        list of floats -> each value represent a fraction total number of columns.
        Note that applies to collums without `active_party_columns` or `shared_columns`.
        They are additionally included in to the partition(s).
    active_party_column : Optional[Union[str, list[str]]]
        Column(s) (typically representing labels) associated with the
        "active party" (which can be the server).
    active_party_columns_mode : Union[Literal[["add_to_first", "add_to_last", "create_as_first", "create_as_last", "add_to_all"], int]
        Determines how to assign the active party columns:
        - "add_to_first": Append active party columns to the first partition.
        - "add_to_last": Append active party columns to the last partition.
        - int: Append active party columns to the specified partition index.
        - "create_as_first": Create a new partition at the start containing only
            these columns.
        - "create_as_last": Create a new partition at the end containing only
            these columns.
        - "add_to_all": Append active party columns to all partitions.
    drop_columns : Optional[list[str]]
        Columns to remove entirely from the dataset before partitioning.
    shared_columns : Optional[list[str]]
        Columns to duplicate into every partition after initial partitioning.
    shuffle : bool
        Whether to shuffle the order of columns before partitioning.
    seed : Optional[int]
        Random seed for shuffling columns. Has no effect if `shuffle=False`.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import VerticalSizePartitioner
    >>>
    >>> partitioner = VerticalSizePartitioner(
    ...     partition_sizes=[8, 4, 2],
    ...     active_party_column="income",
    ...     active_party_columns_mode="create_as_last"
    ... )
    >>> fds = FederatedDataset(
    ...     dataset="scikit-learn/adult-census-income",
    ...     partitioners={"train": partitioner}
    ... )
    >>> partitions = [fds.load_partition(i) for i in range(fds.partitioners["train"].num_partitions)]
    >>> print([partition.column_names for partition in partitions])
    """

    def __init__(
        self,
        partition_sizes: Union[list[int], list[float]],
        active_party_column: Optional[Union[str, list[str]]] = None,
        active_party_columns_mode: Union[
            Literal[
                "add_to_first",
                "add_to_last",
                "create_as_first",
                "create_as_last",
                "add_to_all",
            ],
            int,
        ] = "add_to_last",
        drop_columns: Optional[list[str]] = None,
        shared_columns: Optional[list[str]] = None,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()

        self._partition_sizes = partition_sizes
        self._active_party_columns = self._init_active_party_column(active_party_column)
        self._active_party_columns_mode = active_party_columns_mode
        self._drop_columns = drop_columns or []
        self._shared_columns = shared_columns or []
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        self._partition_columns: Optional[list[list[str]]] = None
        self._partitions_determined = False

        self._validate_parameters_in_init()

    def _determine_partitions_if_needed(self) -> None:
        if self._partitions_determined:
            return

        if self.dataset is None:
            raise ValueError("No dataset is set for this partitioner.")

        all_columns = list(self.dataset.column_names)
        self._validate_parameters_while_partitioning(
            all_columns, self._shared_columns, self._active_party_columns
        )
        columns = [column for column in all_columns if column not in self._drop_columns]
        columns = [column for column in columns if column not in self._shared_columns]
        columns = [
            column for column in columns if column not in self._active_party_columns
        ]

        if self._shuffle:
            self._rng.shuffle(columns)
        if all(isinstance(fraction, float) for fraction in self._partition_sizes):
            partition_columns = _fraction_split(
                columns, cast(list[float], self._partition_sizes)
            )
        else:
            partition_columns = _count_split(
                columns, cast(list[int], self._partition_sizes)
            )

        partition_columns = _add_active_party_columns(
            self._active_party_columns,
            self._active_party_columns_mode,
            partition_columns,
        )

        # Add shared columns to all partitions
        for partition in partition_columns:
            for column in self._shared_columns:
                partition.append(column)

        self._partition_columns = partition_columns
        self._partitions_determined = True

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            Single partition of a dataset.
        """
        self._determine_partitions_if_needed()
        assert self._partition_columns is not None
        if partition_id < 0 or partition_id >= len(self._partition_columns):
            raise IndexError(
                f"partition_id: {partition_id} out of range <0, {self.num_partitions - 1}>."
            )
        columns = self._partition_columns[partition_id]
        return self.dataset.select_columns(columns)

    @property
    def num_partitions(self) -> int:
        """Number of partitions."""
        self._determine_partitions_if_needed()
        assert self._partition_columns is not None
        return len(self._partition_columns)

    def _validate_parameters_in_init(self) -> None:
        if not isinstance(self._partition_sizes, list):
            raise ValueError("partition_sizes must be a list.")
        if all(isinstance(fraction, float) for fraction in self._partition_sizes):
            fraction_sum = sum(self._partition_sizes)
            if fraction_sum != 1.0:
                raise ValueError("Float ratios in `partition_sizes` must sum to 1.0.")
            if any(
                fraction < 0.0 or fraction > 1.0 for fraction in self._partition_sizes
            ):
                raise ValueError(
                    "All floats in `partition_sizes` must be >= 0.0 and <= 1.0."
                )
        elif all(
            isinstance(coulumn_count, int) for coulumn_count in self._partition_sizes
        ):
            if any(coulumn_count < 0 for coulumn_count in self._partition_sizes):
                raise ValueError("All integers in `partition_sizes` must be >= 0.")
        else:
            raise ValueError("`partition_sizes` list must be all floats or all ints.")

        # Validate columns lists
        for parameter_name, parameter_list in [
            ("drop_columns", self._drop_columns),
            ("shared_columns", self._shared_columns),
            ("active_party_columns", self._active_party_columns),
        ]:
            if not all(isinstance(column, str) for column in parameter_list):
                raise ValueError(f"All entries in {parameter_name} must be strings.")

        valid_modes = {
            "add_to_first",
            "add_to_last",
            "create_as_first",
            "create_as_last",
            "add_to_all",
        }
        if not (
            isinstance(self._active_party_columns_mode, int)
            or self._active_party_columns_mode in valid_modes
        ):
            raise ValueError(
                "active_party_columns_mode must be an int or one of "
                "'add_to_first', 'add_to_last', 'create_as_first', 'create_as_last', "
                "'add_to_all'."
            )

    def _validate_parameters_while_partitioning(
        self,
        all_columns: list[str],
        shared_columns: list[str],
        active_party_columns: list[str],
    ) -> None:
        # Shared columns existance check
        for column in shared_columns:
            if column not in all_columns:
                raise ValueError(f"Shared column '{column}' not found in the dataset.")
        # Active party columns existence check
        for column in active_party_columns:
            if column not in all_columns:
                raise ValueError(
                    f"Active party column '{column}' not found in the dataset."
                )
        num_columns = len(all_columns)
        if all(isinstance(size, int) for size in self._partition_sizes):
            if sum(self._partition_sizes) != num_columns:
                raise ValueError(
                    "Sum of partition sizes cannot differ from the total number of columns."
                )
        else:
            pass

    def _init_active_party_column(
        self, active_party_column: Optional[Union[str, list[str]]]
    ) -> list[str]:
        if active_party_column is None:
            return []
        if isinstance(active_party_column, str):
            return [active_party_column]
        if isinstance(active_party_column, list):
            return active_party_column
        raise ValueError("active_party_column must be a string or a list of strings.")


def _count_split(columns: list[str], counts: list[int]) -> list[list[str]]:
    partition_columns = []
    start = 0
    for count in counts:
        end = start + count
        partition_columns.append(columns[start:end])
        start = end
    return partition_columns


def _fraction_split(columns: list[str], fractions: list[float]) -> list[list[str]]:
    num_columns = len(columns)
    partitions = []
    cumulative = 0
    for index, fraction in enumerate(fractions):
        count = int(floor(fraction * num_columns))
        if index == len(fractions) - 1:
            # Last partition takes the remainder
            count = num_columns - cumulative
        partitions.append(columns[cumulative : cumulative + count])
        cumulative += count
    return partitions