"""The methodology to create a Dirichlet distributed Non-IID CIFAR dataset.

obtained from source: https://github.com/JYWa/FedNova/blob/master/util_v4.py
"""

from random import Random

import numpy as np


class Partition:
    """Dataset-like object, but only access a subset of it."""

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        """Return the length of the partition."""
        return len(self.index)

    def __getitem__(self, index):
        """Return the item at index idx."""
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner:
    """Partitions a dataset into different chuncks."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        data,
        sizes,
        seed=2020,
        is_non_iid=False,
        alpha=0,
        dataset=None,
    ):
        self.data = data
        self.dataset = dataset
        if is_non_iid:
            self.partitions, self.ratio = self._get_dirichlet_data_(
                data, sizes, seed, alpha
            )

        else:
            self.partitions = []
            self.ratio = sizes
            rng = Random()
            # rng.seed(seed)
            data_len = len(data)
            indexes = list(range(0, data_len))
            rng.shuffle(indexes)

            for frac in sizes:
                part_len = int(frac * data_len)
                self.partitions.append(indexes[0:part_len])
                indexes = indexes[part_len:]

    def use(self, partition):
        """Return a partition of the dataset."""
        return Partition(self.data, self.partitions[partition])

    @staticmethod
    def _get_dirichlet_data_(
        data, psizes, seed, alpha
    ):  # pylint: disable=too-many-locals
        """Return a partition of the dataset based on Dirichlet distribution."""
        n_nets = len(psizes)
        K = 10
        label_list = np.array(data.targets)
        min_size = 0
        N = len(label_list)
        np.random.seed(seed)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(label_list == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_nets))
                # Balance
                proportions = np.array(
                    [
                        p * (len(idx_j) < N / n_nets)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min((len(idx_j) for idx_j in idx_batch))

        for j in range(n_nets):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}

        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(label_list[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
        # print('Data statistics: %s' % str(net_cls_counts))

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        print("Client Dataset sizes: ", local_sizes)

        return idx_batch, local_sizes
