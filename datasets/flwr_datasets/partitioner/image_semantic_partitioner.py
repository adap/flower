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
"""Image semantic partitioner class that works with Hugging Face Datasets."""

import warnings
from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


# pylint: disable=R0902, R0912, R0914
class ImageSemanticPartitioner(Partitioner):
    """Partitioner based on data semantic information.

    NOTE: Image Semantic Partioner can ONLY work with image dataset.

    This implementation is modified from the original implementation:
    https://github.com/google-research/federated/tree/master/generalization,
    which used tensorflow-federated.

    References：
    What Do We Mean by Generalization in Federated Learning? (accepted by ICLR 2022)
    https://arxiv.org/abs/2110.14216

    (Cited from section 4.1 in the paper)

    Image semantic partitioner's goal is to reverse-engineer the federated
    dataset-generating process so that each client possesses semantically
    similar data. For example, for the EMNIST dataset, we expect every client
    (writer) to (i) write in a consistent style for each digit
    (intra-client intra-label similarity) and (ii) use a consistent writing style
    across all digits  (intra-client inter-label similarity). A simple approach
    might be to cluster similar examples together  and sample client data from
    clusters. However, if one directly clusters the entire dataset, the resulting
    clusters may end up largely correlated to labels. To disentangle the effect
    of label heterogeneity and semantic heterogeneity, we propose the following
    algorithm to enforce intra-client intra-label similarity and intra-client
    inter-label similarity in two separate stages.

    • Stage 1: For each label, we embed examples using a pretrained neural
    network (extracting semantic features), and fit a Gaussian Mixture Model
    to cluster pretrained embeddings into groups. Note that this results
    in multiple groups per label. This stage enforces intra-client
    intra-label consistency.

    • Stage 2: To package the clusters from different labels into clients,
    we aim to compute an optimal multi-partite matching with cost-matrix
    defined by KL-divergence between the Gaussian clusters. To reduce complexity,
    we heuristically solve the optimal multi-partite matching by progressively
    solving the optimal bipartite matching at each time for randomly-chosen
    label pairs. This stage enforces intra-client inter-label consistency.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    efficient_net_type: int
        The type of pretrained EfficientNet model.
        Options: [0, 1, 2, 3, 4, 5, 6, 7], corresponding to EfficientNet B0-B7 models.
    batch_size: int
        The batch size for EfficientNet extracting embeddings.
    pca_components: int
        The number of PCA components for dimensionality reduction.
    gmm_max_iter: int
        The maximum number of iterations for the GMM algorithm.
    gmm_init_params: str
        The initialization method for the GMM algorithm.
        Options: ["random", "kmeans", "k-means++"]
    use_cuda: bool
        Whether to use CUDA for computation acceleration.
    image_column_name: Optional[str]
        The name of the image column in the dataset. If not set, the first image column
        is used.
    kl_pairwise_batch_size: int
        The batch size for computing pairwise KL-divergence of two label clusters.
        Defaults to 32.
    shuffle: bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to partitions.
    rng_seed: Optional[int]
        Seed used for numpy random number generator,
        which used throughout the process. Defaults to None.
    pca_seed: Optional[int]
        Seed used for PCA dimensionality reduction. Defaults to None.
    gmm_seed: Optional[int]
        Seed used for GMM clustering. Defaults to None.


    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import ImageSemanticPartitioner
    >>> partitioner = ImageSemanticPartitioner(
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
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=L
    size=28x28 at 0x7FCF49741B10>, 'label': 3}
    >>> partition_sizes = partition_sizes = [
    >>>     len(fds.load_partition(partition_id)) for partition_id in range(5)
    >>> ]
    >>> print(sorted(partition_sizes))
    [3163, 5278, 5496, 6320, 9522]
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: str,
        efficient_net_type: int = 3,
        batch_size: int = 32,
        pca_components: int = 256,
        gmm_max_iter: int = 100,
        gmm_init_params: str = "random",
        use_cuda: bool = False,
        image_column_name: Optional[str] = None,
        kl_pairwise_batch_size: int = 32,
        shuffle: bool = True,
        rng_seed: Optional[int] = None,
        pca_seed: Optional[int] = None,
        gmm_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._efficient_net_type = efficient_net_type
        self._batch_size = batch_size
        self._efficient_output_size = 1280  # fixed in EfficientNet class
        self._pca_components = pca_components
        self._gmm_max_iter = gmm_max_iter
        self._gmm_init_params = gmm_init_params
        self._use_cuda = use_cuda
        self._image_column_name = image_column_name
        self._kl_pairwise_batch_size = kl_pairwise_batch_size
        self._shuffle = shuffle
        self._rng_seed = rng_seed
        self._pca_seed = pca_seed
        self._gmm_seed = gmm_seed

        self._check_variable_validation()

        self._rng_numpy = np.random.default_rng(seed=self._rng_seed)

        # The attributes below are determined during the first call to load_partition
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
        self._check_data_type_if_needed()
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

    # pylint: disable=C0415, R0915
    def _determine_partition_id_to_indices_if_needed(self) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        try:
            import torch
            from scipy.optimize import linear_sum_assignment
            from sklearn.decomposition import PCA
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
            from torch.distributions import MultivariateNormal, kl_divergence
            from torchvision import models
        except ImportError:
            raise ImportError(
                "ImageSemanticPartitioner requires scikit-learn, torch, "
                "torchvision, scipy, and numpy."
            ) from None
        efficient_nets_dict = [
            (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        ]

        def _pairwise_kl_div(
            means_1: torch.Tensor,
            trils_1: torch.Tensor,
            means_2: torch.Tensor,
            trils_2: torch.Tensor,
            batch_size: int,
            device: torch.device,
        ):
            num_dist_1, num_dist_2 = means_1.shape[0], means_2.shape[0]
            pairwise_kl_matrix = torch.zeros((num_dist_1, num_dist_2), device=device)

            for i in range(0, means_1.shape[0], batch_size):
                for j in range(0, means_2.shape[0], batch_size):
                    pairwise_kl_matrix[i : i + batch_size, j : j + batch_size] = (
                        kl_divergence(
                            MultivariateNormal(
                                means_1[i : i + batch_size].unsqueeze(1),
                                scale_tril=trils_1[i : i + batch_size].unsqueeze(1),
                            ),
                            MultivariateNormal(
                                means_2[j : j + batch_size].unsqueeze(0),
                                scale_tril=trils_2[j : j + batch_size].unsqueeze(0),
                            ),
                        )
                    )
            return pairwise_kl_matrix

        def _subsample(embeddings: NDArrayFloat, num_samples: int) -> NDArrayFloat:
            if len(embeddings) < num_samples:
                return embeddings
            idx_samples = self._rng_numpy.choice(
                len(embeddings), num_samples, replace=False
            )
            return embeddings[idx_samples]  # type: ignore

        backbone, pretrained_weight = efficient_nets_dict[self._efficient_net_type]
        efficient_net: models.EfficientNet = backbone(weights=pretrained_weight)
        efficient_net.classifier = torch.nn.Flatten()

        device = torch.device("cpu")
        if self._use_cuda:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                warnings.warn(
                    "No detected CUDA device, the device fallbacks to CPU.",
                    UserWarning,
                    stacklevel=1,
                )
        efficient_net.to(device)
        efficient_net.eval()

        # Generate information needed for Image semantic partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None

        # Use EfficientNet to extract embeddings
        embedding_list = []
        with torch.no_grad():
            for i in range(0, self.dataset.num_rows, self._batch_size):
                idxs = list(range(i, min(i + self._batch_size, self.dataset.num_rows)))
                images = torch.tensor(
                    self._preprocess_dataset_images(idxs),
                    dtype=torch.float,
                    device=device,
                )
                if images.shape[1] == 1:
                    # due to EfficientNet's input format restraint,
                    # here we need to trans graysacle image to RGB image.
                    images = images.broadcast_to(
                        (images.shape[0], 3, *images.shape[2:])
                    )
                embedding_list.append(efficient_net(images).cpu().numpy())
        embedding_list = np.concatenate(embedding_list)
        embeddings_scaled: NDArrayFloat = StandardScaler(with_std=False).fit_transform(
            embedding_list
        )

        if 0 < self._pca_components < embeddings_scaled.shape[1]:
            pca = PCA(n_components=self._pca_components, random_state=self._pca_seed)
            # 100000 refers to official implementation
            pca.fit(_subsample(embeddings_scaled, 100000))
            embeddings_scaled = pca.transform(embeddings_scaled)

        targets = np.array(self.dataset[self._partition_by], dtype=np.int64)
        label_cluster_means: Dict[int, torch.Tensor] = {}
        label_cluster_trils: Dict[int, torch.Tensor] = {}

        label_cluster_list: List[List[List[int]]] = [
            [[] for _ in range(self._num_partitions)] for _ in self._unique_classes
        ]
        for current_label in self._unique_classes:
            print(f"Buliding clusters of label {current_label}")
            idx_current_label = np.where(targets == current_label)[0]
            # 10000 refers to official implementation
            embeddings_of_current_label = _subsample(
                embeddings_scaled[idx_current_label], 10000
            )

            # Use Gaussian Mixture Model to cluster the embeddings
            gmm = GaussianMixture(
                n_components=self._num_partitions,
                max_iter=self._gmm_max_iter,
                reg_covar=1e-4,
                init_params=self._gmm_init_params,
                random_state=self._gmm_seed,
                verbose=False,
            )

            gmm.fit(embeddings_of_current_label)

            cluster_list = gmm.predict(embeddings_of_current_label)

            for idx, cluster in zip(idx_current_label.tolist(), cluster_list):
                label_cluster_list[current_label][cluster].append(idx)  # type: ignore

            label_cluster_means[current_label] = torch.tensor(  # type: ignore
                gmm.means_, dtype=torch.float, device=device
            )
            label_cluster_trils[current_label] = (
                torch.linalg.cholesky(  # type: ignore
                    torch.from_numpy(gmm.covariances_)
                )
                .float()
                .to(device)
            )

        # Start clustering
        # Format: clusters[i] indicates label i is assigned to clients in clusters[i]
        clusters = [
            [None for _ in range(self._num_partitions)] for _ in self._unique_classes
        ]
        partitions = list(range(self._num_partitions))
        unmatched_labels = list(self._unique_classes)

        latest_matched_label = self._rng_numpy.choice(unmatched_labels)  # type: ignore
        clusters[latest_matched_label] = partitions  # type: ignore

        unmatched_labels.remove(latest_matched_label)

        while unmatched_labels:
            label_to_match = self._rng_numpy.choice(unmatched_labels)  # type: ignore

            print(
                "Computing pairwise KL-divergence between label ",
                f"{latest_matched_label} and {label_to_match}",
            )

            cost_matrix = (
                _pairwise_kl_div(
                    means_1=label_cluster_means[latest_matched_label],
                    trils_1=label_cluster_trils[latest_matched_label],
                    means_2=label_cluster_means[label_to_match],
                    trils_2=label_cluster_trils[label_to_match],
                    batch_size=self._kl_pairwise_batch_size,
                    device=device,
                )
                .cpu()
                .numpy()
            )

            optimal_local_assignment = linear_sum_assignment(cost_matrix)

            for client_id in partitions:
                clusters[label_to_match][optimal_local_assignment[1][client_id]] = (
                    clusters[latest_matched_label][
                        optimal_local_assignment[0][client_id]
                    ]
                )

            unmatched_labels.remove(label_to_match)
            latest_matched_label = label_to_match

        partition_id_to_indices: Dict[int, List[int]] = {i: [] for i in partitions}

        for current_label in self._unique_classes:
            for i in partitions:
                partition_id_to_indices[clusters[current_label][i]].extend(  # type: ignore
                    label_cluster_list[current_label][i]  # type: ignore
                )

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng_numpy.shuffle(indices)
        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    def _preprocess_dataset_images(self, indices: List[int]) -> NDArrayFloat:
        """Preprocess the images in the dataset."""
        images = np.array(self.dataset[indices][self._image_column_name], dtype=float)
        if len(images.shape) == 3:  # [B, H, W]
            images = np.reshape(
                images, (images.shape[0], 1, images.shape[1], images.shape[2])
            )
        elif len(images.shape) == 4:  # 2D
            # [H, W, C]
            if images.shape[3] < min(images.shape[1], images.shape[2]):
                images = np.transpose(images, (0, 3, 1, 2))
            # [C, H, W]
            elif images.shape[1] < min(images.shape[2], images.shape[3]):
                pass
        else:
            raise ValueError(f"The image shape is not supported. Now: {images.shape}")
        return images

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test whether the number of partitions is valid."""
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
                    f"min(the number of samples = {self.dataset.num_rows}, "
                    "efficient net output size = 1280) "
                    "in the dataset or the output size of the efficient net. "
                    f"Now: {self._pca_components}."
                )

    def _check_data_type_if_needed(self) -> None:
        """Test whether data is image-like."""
        if not self._partition_id_to_indices_determined:
            features_dict = self.dataset.features.to_dict()
            if self._image_column_name is None:
                self._image_column_name = list(features_dict.keys())[0]
            if self._image_column_name not in features_dict.keys():
                raise ValueError(
                    "The image column name is not found in the dataset feature dict: ",
                    list(features_dict.keys()),
                    f"Now: {self._image_column_name}. ",
                )
            if not isinstance(
                self.dataset.features[self._image_column_name], datasets.Image
            ):
                warnings.warn(
                    "Image semantic partitioner only supports image-like data. "
                    f"But column '{self._image_column_name}' is not datasets.Image. "
                    "So the partition might be failed.",
                    stacklevel=1,
                )
            try:
                image = np.array(
                    self.dataset[0][self._image_column_name], dtype=np.float32
                )
            except ValueError as err:
                raise ValueError(
                    "The data needs to be able to transform to np.ndarray. "
                ) from err

            if not 2 <= len(image.shape) <= 3:
                raise ValueError(
                    "The image shape is not supported. "
                    "The image shape should among {[H, W], [C, H, W], [H, W, C]}. "
                    f"Now: {image.shape}. "
                )
            if len(image.shape) == 3:
                smallest_axis = min(enumerate(image.shape), key=lambda x: x[1])[0]
                # smallest axis (C) should be at the first or the last place.
                if smallest_axis not in [0, 2]:
                    raise ValueError(
                        "The 3D image shape should be [C, H, W] or [H, W, C]. "
                        f"Now: {image.shape}. "
                    )

    def _check_variable_validation(self) -> None:
        """Test class variables validation."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
        if not (
            isinstance(self._efficient_net_type, int)
            and 0 <= self._efficient_net_type <= 7
        ):
            raise ValueError(
                "The efficient net type needs to be in the range of 0 to 7, "
                "indicates EfficientNet-B0 ~ B7"
            )
        if self._batch_size <= 0:
            raise ValueError("The batch size needs to be greater than zero.")
        if self._gmm_init_params not in ["kmeans", "k-means++", "random"]:
            raise ValueError(
                "The gmm_init_params needs to be in [kmeans, k-means++, random]"
            )
        if self._gmm_max_iter <= 0:
            raise ValueError("The gmm max iter needs to be greater than zero.")
        if self._pca_components <= 0:
            raise ValueError("The pca components needs to be greater than zero.")
        if self._rng_seed and not isinstance(self._rng_seed, int):
            raise TypeError("The rng seed needs to be an integer.")
        if self._pca_seed and not isinstance(self._pca_seed, int):
            raise TypeError("The pca seed needs to be an integer.")
        if self._gmm_seed and not isinstance(self._gmm_seed, int):
            raise TypeError("The gmm seed needs to be an integer.")


if __name__ == "__main__":
    # ===================== Test with custom Dataset =====================
    from datasets import Dataset

    dataset = {
        "image": [np.random.randn(28, 28) for _ in range(50)],
        "label": [i % 3 for i in range(50)],
    }
    dataset = Dataset.from_dict(dataset)
    partitioner = ImageSemanticPartitioner(
        num_partitions=5, partition_by="label", pca_components=30
    )
    partitioner.dataset = dataset
    partition = partitioner.load_partition(0)
    partition_sizes = partition_sizes = [
        len(partitioner.load_partition(partition_id)) for partition_id in range(5)
    ]
    print(sorted(partition_sizes))
