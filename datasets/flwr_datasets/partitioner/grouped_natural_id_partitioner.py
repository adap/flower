import numpy as np

import datasets
from flwr_datasets.partitioner.natural_id_partitioner import NaturalIdPartitioner


class GroupedNaturalIdPartitioner(NaturalIdPartitioner):
    """Partitioner for a dataset in which the final partition results in different non
    overlapping datasets for a given number of groups/nodes.

    Parameters
    ----------
    partition_by : str
        The label in the dataset that divided the data.
    num_groups : int
        The number of groups in which the data will be divided.
    Size:
        
    """

    def __init__(self, partition_by, num_groups=0,group_size=0) -> None:
        super().__init__(partition_by)
        if not group_size and not num_groups:
            raise ValueError("num_groups and group_size cannot both be zero")
        self._num_groups = num_groups
        self._group_size = group_size
        if self._group_size > 0:
            self._MODE = 'GROUP_SIZE'
        else:
            self._MODE = 'NUM_GROUPS'

    def _create_int_node_id_to_natural_id(self) -> None:
        """Create a mapping from int indices to unique client or group ids from dataset.

        Natural ids come from the column specified in `partition_by`.
        """
        num_labels = len(self.dataset.unique(self._partition_by))
        if self._num_groups > num_labels:
            raise ValueError(
                "The number of groups cannot be greater than the number \
                    of labels in the dataset." # Here is where the other partitioner I proposed comes handy
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


        if self._MODE == 'GROUP_SIZE':
            return self.allocate_samples_to_group(node_id)

        if len(self._node_id_to_natural_id) == 0:
            self._create_int_node_id_to_natural_id()

        return self.dataset.filter(
            lambda row: row[self._partition_by] in self._node_id_to_natural_id[node_id]
        )

    def allocate_samples_to_group(self, node_id):
        FLAG = False
        num_groups = len(self.dataset) // self._group_size
        remainder = (len(self.dataset) % self._group_size)

        #If the remainder is too small -> flag to decide what happens
        if remainder <= 0.5 * self._group_size:
            num_groups+= -1
            FLAG = True # FLAG = False to drop

        if node_id > num_groups:
            raise IndexError

        sorted_dataset = self.dataset.sort(self._partition_by) 
        
        start_index = int((node_id) * self._group_size)
        end_index = min(int(node_id * self._group_size + self._group_size),len(self.dataset))
        if FLAG and node_id == num_groups:
            end_index = len(self.dataset)

        allocated_samples = sorted_dataset.select([x for x in range(start_index,end_index)])

        
        return allocated_samples