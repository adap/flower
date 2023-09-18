"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import os
import hydra
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig, OmegaConf

from typing import List, Optional, Tuple, Union
from numpy.random import BitGenerator, Generator, SeedSequence

XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = Tuple[XYList, XYList]

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
    """Print label distribution for list of partitions."""
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
    return nda_adjusted


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return nda_adjusted


def split_array_at_indices(
    x: np.ndarray, split_idx: np.ndarray
) -> List[List[np.ndarray]]:
    """Splits an array `x` into list of elements using starting indices from
    `split_idx`.

        This function should be used with `unique_indices` from `np.unique()` after
        sorting by label.

    Args:
        x (np.ndarray): Original array of dimension (N,a,b,c,...)
        split_idx (np.ndarray): 1-D array contaning increasing number of
            indices to be used as partitions. Initial value must be zero. Last value
            must be less than N.

    Returns:
        List[List[np.ndarray]]: List of list of samples.
    """

    if split_idx.ndim != 1:
        raise ValueError("Variable `split_idx` must be a 1-D numpy array.")
    if split_idx.dtype != np.int64:
        raise ValueError("Variable `split_idx` must be of type np.int64.")
    if split_idx[0] != 0:
        raise ValueError("First value of `split_idx` must be 0.")
    if split_idx[-1] >= x.shape[0]:
        raise ValueError(
            """Last value in `split_idx` must be less than
            the number of samples in `x`."""
        )
    if not np.all(split_idx[:-1] <= split_idx[1:]):
        raise ValueError("Items in `split_idx` must be in increasing order.")

    num_splits: int = len(split_idx)
    split_idx = np.append(split_idx, x.shape[0])

    list_samples_split: List[List[np.ndarray]] = [[] for _ in range(num_splits)]
    for j in range(num_splits):
        tmp_x = x[split_idx[j] : split_idx[j + 1]]  # noqa: E203
        for sample in tmp_x:
            list_samples_split[j].append(sample)

    return list_samples_split


def exclude_classes_and_normalize(
    distribution: np.ndarray, exclude_dims: List[bool], eps: float = 1e-5
) -> np.ndarray:
    """Excludes classes from a distribution.

    This function is particularly useful when sampling without replacement.
    Classes for which no sample is available have their probabilities are set to 0.
    Classes that had probabilities originally set to 0 are incremented with
     `eps` to allow sampling from remaining items.

    Args:
        distribution (np.array): Distribution being used.
        exclude_dims (List[bool]): Dimensions to be excluded.
        eps (float, optional): Small value to be addad to non-excluded dimensions.
            Defaults to 1e-5.

    Returns:
        np.ndarray: Normalized distributions.
    """
    if np.any(distribution < 0) or (not np.isclose(np.sum(distribution), 1.0)):
        raise ValueError("distribution must sum to 1 and have only positive values.")

    if distribution.size != len(exclude_dims):
        raise ValueError(
            """Length of distribution must be equal
            to the length `exclude_dims`."""
        )
    if eps < 0:
        raise ValueError("""The value of `eps` must be positive and small.""")

    distribution[[not x for x in exclude_dims]] += eps
    distribution[exclude_dims] = 0.0
    sum_rows = np.sum(distribution) + np.finfo(float).eps
    distribution = distribution / sum_rows

    return distribution


def sample_without_replacement(
    distribution: np.ndarray,
    list_samples: List[List[np.ndarray]],
    num_samples: int,
    empty_classes: List[bool],
) -> Tuple[XY, List[bool]]:
    """Samples from a list without replacement using a given distribution.

    Args:
        distribution (np.ndarray): Distribution used for sampling.
        list_samples(List[List[np.ndarray]]): List of samples.
        num_samples (int): Total number of items to be sampled.
        empty_classes (List[bool]): List of booleans indicating which classes are empty.
            This is useful to differentiate which classes should still be sampled.

    Returns:
        XY: Dataset contaning samples
        List[bool]: empty_classes.
    """
    if np.sum([len(x) for x in list_samples]) < num_samples:
        raise ValueError(
            """Number of samples in `list_samples` is less than `num_samples`"""
        )

    # Make sure empty classes are not sampled
    # and solves for rare cases where
    if not empty_classes:
        empty_classes = len(distribution) * [False]

    distribution = exclude_classes_and_normalize(
        distribution=distribution, exclude_dims=empty_classes
    )

    data: List[np.ndarray] = []
    target: List[np.ndarray] = []

    for _ in range(num_samples):
        sample_class = np.where(np.random.multinomial(1, distribution) == 1)[0][0]
        sample: np.ndarray = list_samples[sample_class].pop()

        data.append(sample)
        target.append(sample_class)

        # If last sample of the class was drawn, then set the
        #  probability density function (PDF) to zero for that class.
        if len(list_samples[sample_class]) == 0:
            empty_classes[sample_class] = True
            # Be careful to distinguish between classes that had zero probability
            # and classes that are now empty
            distribution = exclude_classes_and_normalize(
                distribution=distribution, exclude_dims=empty_classes
            )
    data_array: np.ndarray = np.concatenate([data], axis=0)
    target_array: np.ndarray = np.array(target, dtype=np.int64)

    return (data_array, target_array), empty_classes


def get_partitions_distributions(partitions: XYList) -> Tuple[np.ndarray, List[int]]:
    """Evaluates the distribution over classes for a set of partitions.

    Args:
        partitions (XYList): Input partitions

    Returns:
        np.ndarray: Distributions of size (num_partitions, num_classes)
    """
    # Get largest available label
    labels = set()
    for _, y in partitions:
        labels.update(set(y))
    list_labels = sorted(list(labels))
    bin_edges = np.arange(len(list_labels) + 1)

    # Pre-allocate distributions
    distributions = np.zeros((len(partitions), len(list_labels)), dtype=np.float32)
    for idx, (_, _y) in enumerate(partitions):
        hist, _ = np.histogram(_y, bin_edges)
        distributions[idx] = hist / hist.sum()

    return distributions, list_labels


def create_lda_partitions(
    dataset: XY,
    dirichlet_dist: Optional[np.ndarray] = None,
    num_partitions: int = 100,
    concentration: Union[float, np.ndarray, List[float]] = 0.5,
    accept_imbalanced: bool = False,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[XYList, np.ndarray]:
    """Create imbalanced non-iid partitions using Latent Dirichlet Allocation
    (LDA) without resampling.

    Args:
        dataset (XY): Dataset containing samples X and labels Y.
        dirichlet_dist (numpy.ndarray, optional): previously generated distribution to
            be used. This is useful when applying the same distribution for train and
            validation sets.
        num_partitions (int, optional): Number of partitions to be created.
            Defaults to 100.
        concentration (float, np.ndarray, List[float]): Dirichlet Concentration
            (:math:`\\alpha`) parameter. Set to float('inf') to get uniform partitions.
            An :math:`\\alpha \\to \\Inf` generates uniform distributions over classes.
            An :math:`\\alpha \\to 0.0` generates one class per client. Defaults to 0.5.
        accept_imbalanced (bool): Whether or not to accept imbalanced output classes.
            Default False.
        seed (None, int, SeedSequence, BitGenerator, Generator):
            A seed to initialize the BitGenerator for generating the Dirichlet
            distribution. This is defined in Numpy's official documentation as follows:
            If None, then fresh, unpredictable entropy will be pulled from the OS.
            One may also pass in a SeedSequence instance.
            Additionally, when passed a BitGenerator, it will be wrapped by Generator.
            If passed a Generator, it will be returned unaltered.
            See official Numpy Documentation for further details.

    Returns:
        Tuple[XYList, numpy.ndarray]: List of XYList containing partitions
            for each dataset and the dirichlet probability density functions.
    """
    # pylint: disable=too-many-arguments,too-many-locals

    x, y = dataset
    x, y = shuffle(x, y)
    x, y = sort_by_label(x, y)

    if (x.shape[0] % num_partitions) and (not accept_imbalanced):
        raise ValueError(
            """Total number of samples must be a multiple of `num_partitions`.
               If imbalanced classes are allowed, set
               `accept_imbalanced=True`."""
        )

    num_samples = num_partitions * [0]
    for j in range(x.shape[0]):
        num_samples[j % num_partitions] += 1

    # Get number of classes and verify if they matching with
    classes, start_indices = np.unique(y, return_index=True)

    # Make sure that concentration is np.array and
    # check if concentration is appropriate
    concentration = np.asarray(concentration)

    # Check if concentration is Inf, if so create uniform partitions
    partitions: List[XY] = [(_, _) for _ in range(num_partitions)]
    if float("inf") in concentration:
        partitions = create_partitions(
            unpartitioned_dataset=(x, y),
            iid_fraction=1.0,
            num_partitions=num_partitions,
        )
        dirichlet_dist = get_partitions_distributions(partitions)[0]

        return partitions, dirichlet_dist

    if concentration.size == 1:
        concentration = np.repeat(concentration, classes.size)
    elif concentration.size != classes.size:  # Sequence
        raise ValueError(
            f"The size of the provided concentration ({concentration.size}) ",
            f"must be either 1 or equal number of classes {classes.size})",
        )

    # Split into list of list of samples per class
    list_samples_per_class: List[List[np.ndarray]] = split_array_at_indices(
        x, start_indices
    )

    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng(seed).dirichlet(
            alpha=concentration, size=num_partitions
        )

    if dirichlet_dist.size != 0:
        if dirichlet_dist.shape != (num_partitions, classes.size):
            raise ValueError(
                f"""The shape of the provided dirichlet distribution
                 ({dirichlet_dist.shape}) must match the provided number
                  of partitions and classes ({num_partitions},{classes.size})"""
            )

    # Assuming balanced distribution
    empty_classes = classes.size * [False]
    for partition_id in range(num_partitions):
        partitions[partition_id], empty_classes = sample_without_replacement(
            distribution=dirichlet_dist[partition_id].copy(),
            list_samples=list_samples_per_class,
            num_samples=num_samples[partition_id],
            empty_classes=empty_classes,
        )

    return partitions, dirichlet_dist


@hydra.main(config_path="conf", config_name="base", version_base=None)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Does everything needed to get the dataset.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """

    ## 1. print parsed config
    print(OmegaConf.to_yaml(cfg))

    dataset = cfg.dataset.dataset
    total_clients = cfg.num_clients

    if dataset in ["fmnist", "cifar10"]:
        num_classes = 10
    else:
        num_classes = 20

    folder = dataset

    # if the folder exists it is deleted and the ds partitions are re-created
    # if the folder does not exist, firstly the folder is created
    # and then the ds partitions are generated
    exist = os.path.exists(folder)
    if not exist:
        os.makedirs(folder)

    if dataset in ["fmnist"]:
        # Load the FMNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif dataset in ["cifar10"]:
        # Load the CIFAR10 dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0  
    
    num_partitions = total_clients
    concentration = cfg.alpha
    seed = cfg.seed if cfg.seed is not None else 42

    # Create the partitions
    partitions, dirichlet_dist = create_lda_partitions(
        dataset=(x_train, y_train),
        num_partitions=num_partitions,
        concentration=concentration,
        seed=seed,
    )

    for cid in range(num_partitions):
        ds = tf.data.Dataset.from_tensor_slices(partitions[cid])
        ds = ds.shuffle(buffer_size=4096)

        # save the dataset partition
        tf.data.experimental.save(ds, os.path.join(folder, str(cid)), compression=None)

    list_of_narrays = []
    for sampled_client in range(0, total_clients):
        loaded_ds = tf.data.experimental.load(
            path=os.path.join(folder, str(sampled_client)), element_spec=None, compression=None, reader_func=None
        )

        print("[Client " + str(sampled_client) + "]")
        print("Cardinality: ", tf.data.experimental.cardinality(loaded_ds).numpy())

        def count_class(counts, batch, num_classes=num_classes):
            _, labels = batch
            for i in range(num_classes):
                cc = tf.cast(labels == i, tf.int32)
                counts[i] += tf.reduce_sum(cc)
            return counts

        initial_state = dict((i, 0) for i in range(num_classes))
        counts = loaded_ds.reduce(initial_state=initial_state, reduce_func=count_class)

        # print([(k, v.numpy()) for k, v in counts.items()])
        new_dict = {k: v.numpy() for k, v in counts.items()}
        # print(new_dict)
        res = np.array([item for item in new_dict.values()])
        # print(res)
        list_of_narrays.append(res)

    distribution = np.stack(list_of_narrays)
    print(distribution)
    # saving the distribution of per-label examples in a numpy file
    # this can be useful also to draw charts about the label distrib.
    path = os.path.join(folder, "distribution_train.npy")
    np.save(path, distribution)

    # Save the x and y together of the test set to disk using numpy
    path = os.path.join(folder, "x_test.npy")
    np.save(path, x_test)
    path = os.path.join(folder, "y_test.npy")
    np.save(path, y_test)



if __name__ == "__main__":

    download_and_preprocess()