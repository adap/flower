# Copyright 2020 Adap GmbH. All Rights Reserved.
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

# pylint: disable=invalid-name

from typing import List, Optional, Tuple, cast

import numpy as np

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = Tuple[XYList, XYList]

np.random.seed(2020)


def float_to_int(i: float) -> int:
    """Return float as int but raise if decimal is dropped."""
    if not i.is_integer():
        raise Exception("Cast would drop decimals")

    return int(i)


def sort_by_label(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort by label.

    Assuming two labels and four examples the resulting label order
    would be 1,1,2,2
    """
    idx = np.argsort(y, axis=0).reshape((y.shape[0]))
    return (x[idx], y[idx])


def sort_by_label_repeating(x: np.ndarray, y: np.ndarray) -> XY:
    """Sort by label in repeating groups. Assuming two labels and four examples
    the resulting label order would be 1,2,1,2.

    Create sorting index which is applied to by label sorted x, y

    .. code-block:: python

        # given:
        y = [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9
        ]

        # use:
        idx = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        ]

        # so that y[idx] becomes:
        y = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ]
    """
    x, y = sort_by_label(x, y)

    num_example = x.shape[0]
    num_class = np.unique(y).shape[0]
    idx = (
        np.array(range(num_example), np.int64)
        .reshape((num_class, num_example // num_class))
        .transpose()
        .reshape(num_example)
    )

    return (x[idx], y[idx])


def split_at_fraction(x: np.ndarray, y: np.ndarray, fraction: float) -> Tuple[XY, XY]:
    """Split x, y at a certain fraction."""
    splitting_index = float_to_int(x.shape[0] * fraction)
    # Take everything BEFORE splitting_index
    x_0, y_0 = x[:splitting_index], y[:splitting_index]
    # Take everything AFTER splitting_index
    x_1, y_1 = x[splitting_index:], y[splitting_index:]
    return (x_0, y_0), (x_1, y_1)


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> List[XY]:
    """Return x, y as list of partitions."""
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def combine_partitions(xy_list_0: XYList, xy_list_1: XYList) -> XYList:
    """Combine two lists of ndarray Tuples into one list."""
    return [
        (np.concatenate([x_0, x_1], axis=0), np.concatenate([y_0, y_1], axis=0))
        for (x_0, y_0), (x_1, y_1) in zip(xy_list_0, xy_list_1)
    ]


def shift(x: np.ndarray, y: np.ndarray) -> XY:
    """Shift x_1, y_1 so that the first half contains only labels 0 to 4 and
    the second half 5 to 9."""
    x, y = sort_by_label(x, y)

    (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction=0.5)
    (x_0, y_0), (x_1, y_1) = shuffle(x_0, y_0), shuffle(x_1, y_1)
    x, y = np.concatenate([x_0, x_1], axis=0), np.concatenate([y_0, y_1], axis=0)
    return x, y


def create_partitions(
    unpartitioned_dataset: XY,
    iid_fraction: float,
    num_partitions: int,
) -> XYList:
    """Create partitioned version of a training or test set.

    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    x, y = unpartitioned_dataset

    x, y = shuffle(x, y)
    x, y = sort_by_label_repeating(x, y)

    (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction=iid_fraction)

    # Shift in second split of dataset the classes into two groups
    x_1, y_1 = shift(x_1, y_1)

    xy_0_partitions = partition(x_0, y_0, num_partitions)
    xy_1_partitions = partition(x_1, y_1, num_partitions)

    xy_partitions = combine_partitions(xy_0_partitions, xy_1_partitions)

    # Adjust x and y shape
    return [adjust_xy_shape(xy) for xy in xy_partitions]


def create_partitioned_dataset(
    keras_dataset: Tuple[XY, XY],
    iid_fraction: float,
    num_partitions: int,
) -> Tuple[PartitionedDataset, XY]:
    """Create partitioned version of keras dataset.

    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    xy_train, xy_test = keras_dataset

    xy_train_partitions = create_partitions(
        unpartitioned_dataset=xy_train,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    xy_test_partitions = create_partitions(
        unpartitioned_dataset=xy_test,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    return (xy_train_partitions, xy_test_partitions), adjust_xy_shape(xy_test)


def log_distribution(xy_partitions: XYList) -> None:
    """Print label distribution for list of paritions."""
    distro = [np.unique(y, return_counts=True) for _, y in xy_partitions]
    for d in distro:
        print(d)


def adjust_xy_shape(xy: XY) -> XY:
    """Adjust shape of both x and y."""
    x, y = xy
    if x.ndim == 3:
        x = adjust_x_shape(x)
    if y.ndim == 2:
        y = adjust_y_shape(y)
    return (x, y)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return cast(np.ndarray, nda_adjusted)


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return cast(np.ndarray, nda_adjusted)


def create_dla_partitions(
    dataset: XY,
    dirichlet_dist: np.ndarray = np.empty(0),
    num_partitions: int = 100,
    concentration: float = 0.5,
) -> Tuple[np.ndarray, XYList]:
    """Create ibalanced non-iid partitions using Dirichlet Latent
    Allocation(LDA) without resampling.

    Args:
        dataset (XY): Datasets containing samples X
            and labels Y.
        dirichlet_dist (numpy.ndarray, optional): previously generated distribution to
            be used. This s useful when applying the same distribution for train and
            validation sets.
        num_partitions (int, optional): Number of partitions to be created.
            Defaults to 100.
        concentration (float, optional): Dirichlet Concentration (:math:`\\alpha`)
            parameter.
            An :math:`\\alpha \\to \\Inf` generates uniform distributions over classes.
            An :math:`\\alpha \\to 0.0` generates on class per client. Defaults to 0.5.

    Returns:
        Tuple[numpy.ndarray, XYList]: List of XYList containing partitions
            for each dataset.
    """

    x, y = dataset
    x, y = shuffle(x, y)
    x, y = sort_by_label(x, y)
    x_l: List[np.ndarray] = list(x)

    # Get number of classes and verify if they matching with
    classes, num_samples_per_class = np.unique(y, return_counts=True)
    num_classes: int = classes.size
    remaining_indices = [j for j in range(num_classes)]

    if dirichlet_dist.size != 0:
        dist_num_partitions, dist_num_classes = dirichlet_dist.shape
        if dist_num_classes != num_classes:
            raise ValueError(
                f"""Number of classes in dataset ({num_classes})
              differs from the one in the provided partitions {dist_num_classes}."""
            )
        if dist_num_partitions != num_partitions:
            raise ValueError(
                f"""The value in `num_partitions` ({num_partitions})
                differs from the one from `dirichlet_dist` {dist_num_partitions}."""
            )

    # Assuming balanced distribution
    num_samples = x.shape[0]
    num_samples_per_partition = num_samples // num_partitions

    boundaries: List[int] = np.append(
        [0], np.cumsum(num_samples_per_class, dtype=np.int)
    )
    list_samples_per_class: List[List[np.ndarray]] = [
        x_l[boundaries[idx] : boundaries[idx + 1]]
        for idx in range(num_classes)  # noqa: E203
    ]

    if dirichlet_dist.size == 0:
        dirichlet_dist = np.random.dirichlet(
            alpha=[concentration] * num_classes, size=num_partitions
        )
    original_dirichlet_dist = dirichlet_dist.copy()

    data: List[List[Optional[np.ndarray]]] = [[] for _ in range(num_partitions)]
    target: List[List[Optional[np.ndarray]]] = [[] for _ in range(num_partitions)]

    for partition_id in range(num_partitions):
        for _ in range(num_samples_per_partition):
            sample_class: int = np.where(
                np.random.multinomial(1, dirichlet_dist[partition_id]) == 1
            )[0][0]
            sample: np.ndarray = list_samples_per_class[sample_class].pop()

            data[partition_id].append(sample)
            target[partition_id].append(sample_class)

            # If last sample of the class was drawn,
            # then set pdf to zero for that class.
            num_samples_per_class[sample_class] -= 1
            if num_samples_per_class[sample_class] == 0:
                remaining_indices.remove(np.where(classes == sample_class)[0][0])
                # Be careful to distinguish between original zero-valued
                # classes and classes that are empty
                dirichlet_dist[:, sample_class] = 0.0
                dirichlet_dist[:, remaining_indices] += 1e-5

                sum_rows = np.sum(dirichlet_dist, axis=1)
                dirichlet_dist = dirichlet_dist / (
                    sum_rows[:, np.newaxis] + np.finfo(float).eps
                )

    partitions = [
        (np.concatenate([data[idx]]), np.concatenate([target[idx]])[..., np.newaxis])
        for idx in range(num_partitions)
    ]

    return partitions, original_dirichlet_dist
