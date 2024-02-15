import datasets
from flwr_datasets.partitioner.group_natural_id_partitioner import (
    GroupNaturalIdPartitioner,
)


class NumNodesGroupedNaturalIdPartitioner(GroupNaturalIdPartitioner):
    """Partitioner for a dataset in which the final partition results in different non
    overlapping datasets for a given number of groups/nodes.

    Parameters
    ----------
    partition_by : str
        The label in the dataset that divided the data.
    num_groups : int
        The number of groups to divide the dataset it
    num_nodes : int
        The number of total nodes.
    """

    def __init__(self, partition_by, num_groups, num_nodes) -> None:
        super().__init__(partition_by, num_groups)
        self._num_nodes = num_nodes

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
        if self._num_nodes <= node_id:
            raise KeyError(
                f"There are only {self._num_nodes} nodes [0-{self._num_nodes - 1}], \
                    {node_id} is out of this range. "
            )

        nodes_per_group = self._num_nodes // self._num_groups
        remainder = self._num_nodes % self._num_groups

        group_sizes = [0] * self._num_groups
        group_positions = [[] for _ in range(self._num_groups)]

        node_index = 0

        for i in range(self._num_groups):
            group_sizes[i] = nodes_per_group
            # Distribute any remaining nodes equally among the first 'remainder' groups
            if remainder > 0:
                group_sizes[i] += 1
                remainder -= 1

            for _ in range(group_sizes[i]):
                group_positions[i].append(node_index)
                node_index += 1

        num_group = 0
        position_in_group = 0
        cond = False
        for i in range(len(group_positions)):
            for j in range(len(group_positions[i])):
                if group_positions[i][j] == node_id:
                    num_group = i
                    position_in_group = j
                    cond = True
                    break
            if cond:
                break

        dataset = super().load_partition(num_group)

        dataset = dataset.shard(
            num_shards=int(group_sizes[num_group]),
            index=position_in_group,
            contiguous=True,
        )

        return dataset
