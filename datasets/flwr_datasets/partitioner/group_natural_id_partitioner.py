import numpy as np

import datasets
from flwr_datasets.partitioner.natural_id_partitioner import NaturalIdPartitioner


class GroupNaturalIdPartitioner(NaturalIdPartitioner):
    """Partitioner for a dataset in which the final partition results in different non
    overlapping datasets for a given number of groups/nodes.

    Parameters
    ----------
    partition_by : str
        The label in the dataset that divided the data.
    num_groups : int
        The number of groups in which the data will be divided.
    """

    def __init__(self, partition_by, num_groups) -> None:
        super().__init__(partition_by)
        self._num_groups = num_groups

    def _create_int_node_id_to_natural_id(self) -> None:
        """Create a mapping from int indices to unique client or group ids from dataset.

        Natural ids come from the column specified in `partition_by`.
        """
        num_labels = len(self.dataset.unique(self._partition_by))
        if self._num_groups > num_labels:
            raise ValueError(
                "The number of groups cannot be greater than the number \
                    of labels in the dataset."
            )
        unique_natural_ids = self.dataset.unique(self._partition_by)

        # Divides the labels between the number of nodes/ groups
        split_natural_ids = np.array_split(unique_natural_ids, self._num_groups)

        self._node_id_to_natural_id = dict(
            zip(range(self._num_groups), split_natural_ids)
        )

    def load_partition(self, node_id: int) -> datasets.Dataset:
        """Load a single partition corresponding to a single `node_id`.

        The choice of the partition is based on the parameter node_id,
        and the mapping computed in the function
        _create_int_node_id_to_natural_id()

        Parameters
        ----------
        node_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        if len(self._node_id_to_natural_id) == 0:
            self._create_int_node_id_to_natural_id()

        return self.dataset.filter(
            lambda row: row[self._partition_by] in self._node_id_to_natural_id[node_id]
        )
