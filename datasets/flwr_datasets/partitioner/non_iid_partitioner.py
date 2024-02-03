from typing import Dict
import datasets
from flwr_datasets.partitioner.natural_id_partitioner import NaturalIdPartitioner
import numpy as np

class NonIidPartitioner(NaturalIdPartitioner):
    """Partitioner for a dataset in which the final partition results in different 
        non overlapping datasets for a given number of groups/nodes"""

    def __init__(self, partition_by, num_nodes) -> None:
        super().__init__(partition_by)
        self._num_nodes = num_nodes

    def _create_int_node_id_to_natural_id(self) -> None:
        """Create a mapping from int indices to unique client or group ids from dataset.

        Natural ids come from the column specified in `partition_by`.
        """
        unique_natural_ids = self.dataset.unique(self._partition_by)

        # Divides the labels between the number of nodes/ groups
        split_natural_ids = np.array_split(unique_natural_ids, self._num_nodes)

        self._node_id_to_natural_id = dict(
            zip(range(self._num_nodes), split_natural_ids)
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
    
    

    @property
    def node_id_to_natural_id(self) -> Dict[int, str]:
        """Node id to corresponding natural id present.

        Natural ids are the unique values in `partition_by` column in dataset.
        """
        return self._node_id_to_natural_id

    # pylint: disable=R0201
    @node_id_to_natural_id.setter
    def node_id_to_natural_id(self, value: Dict[int, str]) -> None:
        raise AttributeError(
            "Setting the node_id_to_natural_id dictionary is not allowed."
        )

