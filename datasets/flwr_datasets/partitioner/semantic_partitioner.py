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
"""Semantic partitioner class that works with Hugging Face Datasets."""
# NOTE: Semantic Partioner can only work with image dataset.

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import numpy as np
import datasets
from torch.distributions import MultivariateNormal, kl_divergence
from torchvision import models
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


# pylint: disable=R0902, R0912, R0914
class SemanticPartitioner(Partitioner):
    """Partitioner based on data semantic information.

    Implementation based on Bayesian Nonparametric Federated Learning of Neural Networks
    https://arxiv.org/abs/1905.12022.

    The algorithm sequentially divides the data with each label. The fractions of the
    data with each label is drawn from Dirichlet distribution and adjusted in case of
    balancing. The data is assigned. In case the `min_partition_size` is not satisfied
    the algorithm is run again (the fractions will change since it is a random process
    even though the alpha stays the same).

    The notion of balancing is explicitly introduced here (not mentioned in paper but
    implemented in the code). It is a mechanism that excludes the partition from
    assigning new samples to it if the current number of samples on that partition
    exceeds the average number that the partition would get in case of even data
    distribution. It is controlled by`self_balancing` parameter.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    alpha : Union[int, float, List[float], NDArrayFloat]
        Concentration parameter to the Dirichlet distribution
    min_partition_size : int
        The minimum number of samples that each partitions will have (the sampling
        process is repeated if any partition is too small).
    self_balancing : bool
        Whether assign further samples to a partition after the number of samples
        exceeded the average number of samples per partition. (True in the original
        paper's code although not mentioned in paper itself).
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to partitions.
    seed: int
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import SemanticPartitioner
    >>>
    >>> partitioner = SemanticPartitioner(
    >>>    num_partitions=5, partition_by="label", gmm_max_iter=2
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7FE9D07D2C20>, 'label': 9}
    >>> partition_sizes = partition_sizes = [
    >>>     len(fds.load_partition(partition_id)) for partition_id in range(5)
    >>> ]
    >>> print(sorted(partition_sizes))
    [8660, 8751, 13120, 13672, 15797]
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: str,
        efficient_net_type: int = 0,
        pca_components: int = 128,
        gmm_max_iter: int = 100,
        gmm_init_params: str = "kmeans",
        use_cuda: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        _efficient_nets_dict = [
            (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        ]
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._efficient_net_type = efficient_net_type
        self._efficient_output_size = 1280  # fixed in EfficientNet class
        self._pca_components = pca_components
        self._gmm_max_iter = gmm_max_iter
        self._gmm_init_params = gmm_init_params
        self._use_cuda = use_cuda
        self._shuffle = shuffle
        self._seed = seed
        self._rng_numpy = np.random.default_rng(seed=self._seed)
        self._check_variable_validation()
        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._efficient_net_backbone: Callable[[Any], models.EfficientNet] = (
            _efficient_nets_dict[self._efficient_net_type][0]
        )
        self._efficient_net_pretrained_weight: models.WeightsEnum = (
            _efficient_nets_dict[self._efficient_net_type][1]
        )
        self._unique_classes: Optional[Union[List[int], List[str]]] = None
        self._partition_id_to_indices: Dict[int, List[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single partition of a dataset
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._check_pca_components_validation_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _subsample(self, embeddings: NDArrayFloat, num_samples: int):
        if len(embeddings) < num_samples:
            return embeddings
        idx_samples = self._rng_numpy.choice(
            len(embeddings), num_samples, replace=False
        )
        return embeddings[idx_samples]

    def _determine_partition_id_to_indices_if_needed(self) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        efficient_net: models.EfficientNet = self._efficient_net_backbone(
            weights=self._efficient_net_pretrained_weight
        )
        efficient_net.classifier = torch.nn.Flatten()
        device = torch.device("cpu")
        if self._use_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                warnings("No detected CUDA device, the device fallbacks to CPU.")
        efficient_net.to(device)
        efficient_net.eval()

        # Generate information needed for Semantic partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None

        # Change targets list data type to numpy
        batch_size = 8
        images = self._preprocess_dataset_images()
        embeddings = []
        with torch.no_grad():
            for i in range(0, images.shape[0], batch_size):
                x = torch.tensor(
                    images[i : i + batch_size], dtype=torch.float, device=device
                )
                if x.shape[1] == 1:
                    x = x.broadcast_to((x.shape[0], 3, *x.shape[2:]))
                embeddings.append(efficient_net(x).cpu().numpy())
        embeddings = np.concatenate(embeddings)
        embeddings = StandardScaler(with_std=False).fit_transform(embeddings)

        if 0 < self._pca_components < embeddings.shape[1]:
            pca = PCA(n_components=self._pca_components, random_state=self._seed)
            pca.fit(self._subsample(embeddings, 100000))

        targets = np.array(self.dataset[self._partition_by], dtype=np.int64)
        label_cluster_means = [None for _ in self._unique_classes]
        label_cluster_trils = [None for _ in self._unique_classes]

        gmm = GaussianMixture(
            n_components=self._num_partitions,
            max_iter=self._gmm_max_iter,
            reg_covar=1e-4,
            init_params=self._gmm_init_params,
            random_state=self._seed,
        )

        label_cluster_list = [
            [[] for _ in range(self._num_partitions)] for _ in self._unique_classes
        ]
        for label in self._unique_classes:

            idx_current_label = np.where(targets == label)[0]
            embeddings_of_current_label = self._subsample(
                embeddings[idx_current_label], 10000
            )

            gmm.fit(embeddings_of_current_label)

            cluster_list = gmm.predict(embeddings_of_current_label)

            for idx, cluster in zip(idx_current_label.tolist(), cluster_list):
                label_cluster_list[label][cluster].append(idx)

            label_cluster_means[label] = torch.tensor(gmm.means_)
            label_cluster_trils[label] = torch.linalg.cholesky(
                torch.from_numpy(gmm.covariances_)
            )

        cluster_assignment = [
            [None for _ in range(self._num_partitions)] for _ in self._unique_classes
        ]
        partitions = list(range(self._num_partitions))
        unmatched_labels = list(self._unique_classes)

        latest_matched_label = self._rng_numpy.choice(unmatched_labels)
        cluster_assignment[latest_matched_label] = partitions

        unmatched_labels.remove(latest_matched_label)

        while unmatched_labels:
            label_to_match = self._rng_numpy.choice(unmatched_labels)

            cost_matrix = (
                _pairwise_kl_div(
                    means_1=label_cluster_means[latest_matched_label],
                    trils_1=label_cluster_trils[latest_matched_label],
                    means_2=label_cluster_means[label_to_match],
                    trils_2=label_cluster_trils[label_to_match],
                    device=device,
                )
                .cpu()
                .numpy()
            )

            optimal_local_assignment = linear_sum_assignment(cost_matrix)

            for client_id in partitions:
                cluster_assignment[label_to_match][
                    optimal_local_assignment[1][client_id]
                ] = cluster_assignment[latest_matched_label][
                    optimal_local_assignment[0][client_id]
                ]

            unmatched_labels.remove(label_to_match)
            latest_matched_label = label_to_match

        partition_id_to_indices: Dict[int, List[int]] = {i: [] for i in partitions}

        for label in self._unique_classes:
            for partition_id in partitions:
                partition_id_to_indices[cluster_assignment[label][partition_id]].extend(
                    label_cluster_list[label][partition_id]
                )

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng_numpy.shuffle(indices)
        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    def _preprocess_dataset_images(self):
        images = np.array(self.dataset["image"], dtype=np.float32)
        if len(images.shape) == 3:  # 1D
            images = np.reshape(
                images, (images.shape[0], 1, images.shape[1], images.shape[2])
            )
        elif len(images.shape) == 4:  # 2D
            images = np.transpose(images, (0, 3, 1, 2))
        else:
            raise ValueError("The image shape is not supported.")
        return images

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_pca_components_validation_if_needed(self) -> None:
        """Test whether pca_components is in the valid range."""
        if not self._partition_id_to_indices_determined:

            if self._pca_components > min(
                self.dataset.num_rows, self._efficient_output_size
            ):
                raise ValueError(
                    "The pca_components needs to be smaller than "
                    f"min(the number of samples = {self.dataset.num_rows}, efficient net output size = 1280) "
                    "in the dataset or the output size of the efficient net. "
                    f"Now: {self._pca_components}."
                )

    def _check_variable_validation(self):
        """Test class variables validation."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
        if not (0 <= self._efficient_net_type < 8):
            raise ValueError(
                "The efficient net type needs to be in the range of 0 to 7, indicates EfficientNet-B0 ~ B7"
            )
        if self._gmm_init_params not in ["kmeans", "k-means++", "random"]:
            raise ValueError(
                "The gmm_init_params needs to be in [kmeans, k-means++, random]"
            )
        if self._gmm_max_iter <= 0:
            raise ValueError("The gmm max iter needs to be greater than zero.")
        if self._pca_components <= 0:
            raise ValueError("The pca components needs to be greater than zero.")


def _pairwise_kl_div(
    means_1: torch.Tensor,
    trils_1: torch.Tensor,
    means_2: torch.Tensor,
    trils_2: torch.Tensor,
    device: torch.device,
):
    num_dist_1, num_dist_2 = means_1.shape[0], means_2.shape[0]
    pairwise_kl_matrix = torch.zeros((num_dist_1, num_dist_2), device=device)

    for i in range(means_1.shape[0]):
        for j in range(means_2.shape[0]):
            pairwise_kl_matrix[i, j] = kl_divergence(
                MultivariateNormal(means_1[i], scale_tril=trils_1[i]),
                MultivariateNormal(means_2[j], scale_tril=trils_2[j]),
            )
    return pairwise_kl_matrix


if __name__ == "__main__":
    from flwr_datasets import FederatedDataset
    from datasets import Dataset

    # data = {
    #     "labels": [i % 3 for i in range(50)],
    #     "features": [np.random.randn(1, 28, 28) for _ in range(50)],
    # }
    # dataset = Dataset.from_dict(data)
    partitioner = SemanticPartitioner(
        num_partitions=5, partition_by="label", gmm_max_iter=2
    )
    # partitioner.dataset = dataset
    # partitioner.load_partition(0)
    fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    partition = fds.load_partition(0)
    print(partition[0])  # Print the first example
    partition_sizes = partition_sizes = [
        len(fds.load_partition(partition_id)) for partition_id in range(5)
    ]
    print(sorted(partition_sizes))
