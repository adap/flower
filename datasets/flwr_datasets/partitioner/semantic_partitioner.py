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


import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.distributions import MultivariateNormal, kl_divergence
from torchvision import models

import datasets
from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


# pylint: disable=R0902, R0912, R0914
class SemanticPartitioner(Partitioner):
    """Partitioner based on data semantic information.

    NOTE: Semantic Partioner can ONLY work with image dataset.

    This implementation is modified from the original implementation:
    https://github.com/google-research/federated/tree/master/generalization,
    which used tensorflow-federated.

    References：
    What Do We Mean by Generalization in Federated Learning? (accepted by ICLR 2022)
    https://arxiv.org/abs/2110.14216

    (Cited from section 4.1 in the paper)

    Semantic partitioner's goal is to reverse-engineer the federated dataset-generating
    process so that each client possesses semantically similar data. For example, for
    the EMNIST dataset, we expect every client (writer) to (i) write in a consistent style
    for each digit  (intra-client intra-label similarity) and (ii) use a consistent writing
    style across all digits  (intra-client inter-label similarity). A simple approach might
    be to cluster similar examples together  and sample client data from clusters. However,
    if one directly clusters the entire dataset,  the resulting clusters may end up largely
    correlated to labels. To disentangle the effect of label  heterogeneity and semantic
    heterogeneity, we propose the following algorithm to enforce  intra-client intra-label
    similarity and intra-client inter-label similarity in two separate stages.

    • Stage 1: For each label, we embed examples using a pretrained neural network
    (extracting semantic features), and fit a Gaussian Mixture Model to cluster pretrained
    embeddings into groups. Note that this results in multiple groups per label.
    This stage enforces intra-client intra-label consistency.

    • Stage 2: To package the clusters from different labels into clients, we aim to compute
    an optimal multi-partite matching with cost-matrix defined by KL-divergence between
    the Gaussian clusters. To reduce complexity, we heuristically solve the optimal multi-partite
    matching by progressively solving the optimal bipartite matching at each time for
    randomly-chosen label pairs.
    This stage enforces intra-client inter-label consistency.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    efficient_net_type: int
        The type of pretrained EfficientNet model.
        Options: [0, 1, 2, 3, 4, 5, 6, 7], corresponding to EfficientNet B0-B7 models.
    pca_components: int
        The number of PCA components for dimensionality reduction.
    gmm_max_iter: int
        The maximum number of iterations for the GMM algorithm.
    gmm_init_params: str
        The initialization method for the GMM algorithm.
        Options: ["random", "kmeans", "k-means++"]
    use_cuda: bool
        Whether to use CUDA for computation acceleration.
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to partitions.
    seed: int
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import SemanticPartitioner
    >>> partitioner = SemanticPartitioner(
    >>>     num_partitions=10,
    >>>     partition_by="label",
    >>>     pca_components=128,
    >>>     gmm_max_iter=100,
    >>>     gmm_init_params="kmeans",
    >>>     use_cuda=True,
    >>>     shuffle=True,
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    >>> {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x7FCF49741B10>, 'label': 3}
    >>> partition_sizes = partition_sizes = [
    >>>     len(fds.load_partition(partition_id)) for partition_id in range(5)
    >>> ]
    >>> print(sorted(partition_sizes))
    >>> [3163, 5278, 5496, 6320, 9522]
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: str,
        efficient_net_type: int = 3,
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
        # defaults, but some datasets have different names, e.g. cifar10 is "img"
        # So this variable might be changed in self._check_dataset_type_if_needed()
        self._data_column_name = "image"
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
        self._check_data_validation_if_needed()
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
        images = np.array(self.dataset[self._data_column_name], dtype=np.float32)
        if len(images.shape) == 3:  # 1D
            images = np.reshape(
                images, (images.shape[0], 1, images.shape[1], images.shape[2])
            )
        elif len(images.shape) == 4:  # 2D
            x, y, z = images.shape[1:]
            if z < x and z < y:  # [H, W, C]
                images = np.transpose(images, (0, 3, 1, 2))
            elif x < y and x < z:  # [C, H, W]
                pass
        else:
            raise ValueError(f"The image shape is not supported. Now: {images.shape}")
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

    def _check_data_validation_if_needed(self):
        """Test whether dataset is image dataset"""
        if not self._partition_id_to_indices_determined:
            features_dict = self.dataset.features.to_dict()
            self._data_column_name = list(features_dict.keys())[0]
            try:
                data = np.array(
                    self.dataset[self._data_column_name][0], dtype=np.float32
                )
            except:
                raise TypeError(
                    "The dataset needs to be image dataset. "
                    f"Now: {type(self.dataset[self._data_column_name][0])}."
                )

            if not (2 <= len(data.shape) <= 3):
                raise ValueError(
                    "The image shape is not supported. "
                    "The image shape should among {[H, W], [C, H, W], [H, W, C]}. "
                    f"Now: {data.shape}. "
                )
            elif len(data.shape) == 3:
                x, y, z = data.shape
                if not ((x < y and x < z) or (z < x and z < y)):
                    raise ValueError(
                        "The 3D image shape should be [C, H, W] or [H, W, C]. "
                        f"Now: {data.shape}. "
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
    # ===================== Test with custom Dataset =====================
    from datasets import Dataset

    data = {
        "image": [np.random.randn(28, 28) for _ in range(50)],
        "label": [i % 3 for i in range(50)],
    }
    dataset = Dataset.from_dict(data)
    partitioner = SemanticPartitioner(
        num_partitions=5, partition_by="label", pca_components=30
    )
    partitioner.dataset = dataset
    partition = partitioner.load_partition(0)
    partition_sizes = partition_sizes = [
        len(partitioner.load_partition(partition_id)) for partition_id in range(5)
    ]
    print(sorted(partition_sizes))
    # ====================================================================

    # ===================== Test with FederatedDataset =====================
    # from flwr_datasets import FederatedDataset
    # partitioner = SemanticPartitioner(
    #     num_partitions=5, partition_by="label", pca_components=128
    # )
    # fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    # partition = fds.load_partition(0)
    # print(partition[0])  # Print the first example
    # partition_sizes = partition_sizes = [
    #     len(fds.load_partition(partition_id)) for partition_id in range(5)
    # ]
    # print(sorted(partition_sizes))
    # ======================================================================
