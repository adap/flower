"""Required imports for data.py script."""

import os

import numpy as np
import pandas as pd
import tensorflow as tf

from fedstar.utils import AudioTools, DataTools


def ambient_context_path_extracter(path):
    """Extract and construct a file path for ambient context from a given path string.

    This function takes a path string, splits it at underscores,
    removes the last segment, and then reconstructs the path by
    re-joining the segments and appending the original
    path at the end, separated by the OS-specific directory separator.

    Parameters
    ----------
    - path (str): The original path string to be processed.

    Returns
    -------
    - str: The reconstructed file path suitable for ambient context usage.

    Example:
    Given a path string 'dir_subdir_file_xyz', the function will return
    'dir_subdir/dir_subdir_file_xyz'.
    """
    arr = path.split("_")[:-1]
    file_path = "_".join(arr) + os.sep + path
    return file_path


class DataBuilder:
    """A utility class for building and processing datasets for audio analysis.

    This class provides methods to load, split, and prepare datasets for training
    and testing in audio-related machine learning tasks. It supports operations like
    loading data from files, splitting datasets into labelled and unlabelled sets,
    balancing datasets, and converting them into TensorFlow dataset objects.

    Class Attributes:
    - AUTOTUNE: TensorFlow's setting for optimized parallel data loading.
    - WORDS: List of specific words used in the audio processing context.

    Methods are designed to work with specific data structures and assume a certain
    directory and file naming convention.
    """

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    WORDS = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes",
        "_silence_",
    ]

    # pylint: disable=too-many-arguments, too-many-locals, too-many-statements
    @staticmethod
    def get_files(parent_path, data_dir, train=False, raw=False):
        """Load audio file paths and labels from specified directories.

        Reads training or testing data file paths and their corresponding labels from
        given directories. Supports different dataset types like 'speech_commands' and
        'ambient_context'. Can return raw file paths and labels or a TensorFlow dataset.

        Parameters
        ----------
        - parent_path (str): Base path for the dataset directories.
        - data_dir (str): Specific directory name containing the dataset.
        - train (bool): If True, load training data; otherwise, load testing data.
        - raw (bool): If True, return raw file paths and labels; otherwise, return a
        TensorFlow dataset.

        Returns
        -------
        - A TensorFlow dataset or a tuple of file paths and labels,
        along with the number of classes in the dataset.
        """
        path = os.path.join(parent_path, "data_splits", data_dir)
        train_path = os.path.join(path, "train_split.txt")
        test_path = os.path.join(path, "test_split.txt")
        path_data_dir = os.path.join(parent_path, "datasets", data_dir)
        if train:
            train_files_path, train_labels = [], []
            if data_dir == "speech_commands":
                print("Dataset is speech_commands")
                train_user_files = pd.read_csv(train_path, header=None).values.flatten()
                for tr_uf in train_user_files:
                    path_tr_uf = os.path.join(*tr_uf.split("/"))
                    train_files_path.append(
                        os.path.join(path_data_dir, "Data", "Train", path_tr_uf)
                    )
                    label = tr_uf.split("/")[0]
                    if label in DataBuilder.WORDS:
                        train_labels.append(tr_uf.split("/")[0])
                    else:
                        train_labels.append("_unknown_")
            elif data_dir == "ambient_context":
                print("Dataset is ambient_context")
                train_user_files = pd.read_csv(train_path, sep="\t", header=None).values
                for path, label in train_user_files:
                    file_path = ambient_context_path_extracter(path)
                    train_files_path.append(
                        os.path.join(path_data_dir, "Data", file_path)
                    )
                    train_labels.append(label)
            # One-hot labels transformation
            train_labels = np.array(train_labels, dtype=object)
            # Map labels to 0-11
            unique_labels = np.unique(train_labels)
            num_classes = len(unique_labels)
            labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
            train_labels = [
                labels_dict[train_labels[i]] for i in range(len(train_labels))
            ]
            # Change from object type to int
            train_labels = np.array(train_labels, dtype=np.int64)
            # Convert to dataset (if necessary)
            dataset = (
                (train_files_path, train_labels)
                if raw
                else tf.data.Dataset.from_tensor_slices(
                    (train_files_path, train_labels)
                ).map(AudioTools.read_audio, num_parallel_calls=DataBuilder.AUTOTUNE)
            )
        else:
            # Read files from txt file.
            test_files_path, test_labels = [], []
            if data_dir == "speech_commands":
                print("Dataset is speech_commands")
                test_user_files = pd.read_csv(test_path, header=None).values.flatten()
                for ts_uf in test_user_files:
                    path_ts_uf = os.path.join(*ts_uf.split("/"))
                    test_files_path.append(
                        os.path.join(path_data_dir, "Data", "Test", path_ts_uf)
                    )
                    test_labels.append(ts_uf.split("/")[0])
            elif data_dir == "ambient_context":
                print("Dataset is ambient_context")
                test_user_files = pd.read_csv(test_path, sep="\t", header=None).values
                for path, label in test_user_files:
                    file_path = ambient_context_path_extracter(path)
                    test_files_path.append(
                        os.path.join(path_data_dir, "Data", file_path)
                    )
                    test_labels.append(label)
            # One-hot labels transformation
            test_labels = np.array(test_labels, dtype=object)
            # Map labels to 0-12
            unique_labels = np.unique(test_labels)
            num_classes = len(unique_labels)
            labels_dict = {unique_labels[i]: i for i in range(len(unique_labels))}
            test_labels = [labels_dict[test_labels[i]] for i in range(len(test_labels))]
            # Change from object type to int
            test_labels = np.array(test_labels, dtype=np.int64)
            # Convert to dataset (if necessary)
            dataset = (
                (test_files_path, test_labels)
                if raw
                else tf.data.Dataset.from_tensor_slices(
                    (test_files_path, test_labels)
                ).map(AudioTools.read_audio, num_parallel_calls=DataBuilder.AUTOTUNE)
            )
        return dataset, num_classes

    # pylint: disable=too-many-arguments, too-many-locals
    @staticmethod
    def split_dataset(
        parent_path,
        data_dir,
        num_clients,
        client,
        batch_size=64,
        variance=0.25,
        l_per=0.1,
        u_per=1.0,
        mean_class_distribution=3,
        class_distribute=False,
        fedstar=False,
        seed=2021,
    ):
        """Split dataset into labelled and unlabelled sets for federated learning.

        This method divides the dataset into portions suitable for training by multiple
        clients in a federated learning context. It handles both labelled and unlabelled
        data, and supports class-based and sample-based distribution strategies.

        Parameters
        ----------
        - parent_path (str): Base path for the dataset directories.
        - data_dir (str): Specific directory name containing the dataset.
        - num_clients (int): Total number of clients in the federated learning setup.
        - client (int): Identifier for the current client.
        - batch_size (int): Size of the batches for training.
        - variance (float): Variance allowed in the distribution of samples/classes.
        - l_per (float): Percentage of labelled data to be used.
        - u_per (float): Percentage of unlabelled data to be used.
        - mean_class_distribution (int): Average number of classes per client.
        - class_distribute (bool): If True, distribute data based on classes.
        - fedstar (bool): If True, apply federated star learning setup.
        - seed (int): Random seed for reproducibility.

        Returns
        -------
        - Labelled and unlabelled TensorFlow datasets, number of classes, and number
        of batches for the specified client.
        """
        # Load data
        ds_train, num_classes = DataBuilder.get_files(
            parent_path=parent_path, data_dir=data_dir, train=True, raw=True
        )
        (ds_train_labelled, labelled_size), (
            ds_train_un_labelled,
            unlabelled_size,
        ) = DataTools.get_subset(
            dataset=ds_train,
            percentage=l_per,
            u_per=u_per,
            num_classes=num_classes,
            seed=seed,
        )
        ds_train_un_labelled = (
            DataTools.convert_to_unlabelled(
                dataset=ds_train_un_labelled, unlabelled_data_identifier=-1
            )
            if fedstar
            else []
        )
        # Split data
        if not class_distribute:  # Split according to number of samples
            labelled_sets = list(
                DataTools.distribute_per_samples(
                    dataset=ds_train_labelled,
                    num_clients=num_clients,
                    variance=variance,
                    seed=seed,
                )
            )
        else:  # Split according to classes
            labelled_sets = DataTools.distribute_per_class_with_class_limit(
                dataset=ds_train_labelled,
                num_clients=num_clients,
                num_classes=num_classes,
                mean_class_distribution=mean_class_distribution,
                class_variance=variance,
                seed=seed,
            )
        unlabelled_sets = (
            list(
                DataTools.distribute_per_samples(
                    dataset=ds_train_un_labelled,
                    num_clients=num_clients,
                    variance=variance,
                    seed=seed,
                )
            )
            if fedstar
            else ([], [])
        )
        # Convert to tf dataset objects
        ds_train_labelled = tf.data.Dataset.from_tensor_slices(
            (labelled_sets[client][0], labelled_sets[client][1])
        ).map(AudioTools.read_audio, num_parallel_calls=DataBuilder.AUTOTUNE)
        ds_train_unlabelled = (
            tf.data.Dataset.from_tensor_slices(
                (unlabelled_sets[client][0], unlabelled_sets[client][1])
            ).map(AudioTools.read_audio, num_parallel_calls=DataBuilder.AUTOTUNE)
            if fedstar
            else None
        )
        # Calculate datasets info for training
        labelled_size, unlabelled_size = (
            len(labelled_sets[client][0]),
            len(unlabelled_sets[client][0]) if fedstar else 0,
        )
        num_batches = (
            (unlabelled_size + batch_size - 1) // batch_size
            if fedstar
            else (labelled_size + batch_size - 1) // batch_size
        )
        # Print datasets sizes
        print(
            f"""Client {client}: Train data {labelled_size+unlabelled_size}
            (Unlabelled: {unlabelled_size} - Labelled: {labelled_size})"""
        )
        return ds_train_labelled, ds_train_unlabelled, num_classes, num_batches

    @staticmethod
    def get_ds_test(parent_path, data_dir, batch_size, buffer=1024, seed=2021):
        """Load and prepare the testing dataset.

        Reads testing data from specified paths, converts it into a TensorFlow dataset,
        and prepares it for model evaluation or testing.

        Parameters
        ----------
        - parent_path (str): Base path for the dataset directories.
        - data_dir (str): Specific directory name containing the testing dataset.
        - batch_size (int): Size of the batches for testing.
        - buffer (int): Buffer size for shuffling the dataset.
        - seed (int): Random seed for reproducibility.

        Returns
        -------
        - A TensorFlow dataset for testing and the number of classes in the dataset.
        """
        ds_test, num_classes = DataBuilder.get_files(
            parent_path=parent_path, data_dir=data_dir
        )
        _, _, ds_test = DataBuilder.to_dataset(
            ds_train_labelled=None,
            ds_train_un_labelled=None,
            ds_test=ds_test,
            buffer=buffer,
            batch_size=batch_size,
            seed=seed,
        )
        return ds_test, num_classes

    # pylint: disable=too-many-arguments, too-many-locals
    @staticmethod
    def load_sharded_dataset(
        parent_path,
        data_dir,
        num_clients,
        client,
        variance=0.25,
        batch_size=64,
        l_per=1.0,
        u_per=1.0,
        mean_class_distribution=5,
        fedstar=False,
        class_distribute=False,
        balance_dataset: bool = False,
        seed=2021,
    ):
        """Load and shard the dataset for federated learning scenarios.

        This method prepares a dataset specifically sharded for a given client in a
        federated learning setup. It supports configurations for balanced and unbalanced
        datasets, federated star learning, and class-based distribution.

        Parameters
        ----------
        - parent_path (str): Base path for the dataset directories.
        - data_dir (str): Specific directory name containing the dataset.
        - num_clients (int): Total number of clients in the federated learning setup.
        - client (int): Identifier for the current client.
        - variance (float): Variance allowed in the distribution of samples/classes.
        - batch_size (int): Size of the batches for training.
        - l_per (float): Percentage of labelled data to be used.
        - u_per (float): Percentage of unlabelled data to be used.
        - mean_class_distribution (int): Average number of classes per client.
        - fedstar (bool): If True, apply federated star learning setup.
        - class_distribute (bool): If True, distribute data based on classes.
        - balance_dataset (bool): If True, balance the dataset for each class.
        - seed (int): Random seed for reproducibility.

        Returns
        -------
        - Labelled and unlabelled TensorFlow datasets, number of classes, and number
        of batches for the specified client.
        """
        batch_size = batch_size // 2 if fedstar else batch_size
        (
            ds_train_labelled,
            ds_train_un_labelled,
            num_classes,
            num_batches,
        ) = DataBuilder.split_dataset(
            parent_path=parent_path,
            data_dir=data_dir,
            client=client,
            num_clients=num_clients,
            l_per=l_per,
            u_per=u_per,
            fedstar=fedstar,
            class_distribute=class_distribute,
            mean_class_distribution=mean_class_distribution,
            batch_size=batch_size,
            variance=variance,
            seed=seed,
        )
        ds_train_labelled, ds_train_un_labelled, _ = DataBuilder.to_dataset(
            ds_train_labelled=ds_train_labelled,
            ds_train_un_labelled=ds_train_un_labelled,
            ds_test=None,
            seed=seed,
            batch_size=batch_size,
            balance_dataset=balance_dataset,
        )
        return ds_train_labelled, ds_train_un_labelled, num_classes, num_batches

    # pylint: disable=too-many-arguments, too-many-locals
    @staticmethod
    def to_dataset(
        ds_train_labelled,
        ds_train_un_labelled,
        ds_test,
        batch_size,
        buffer=1024,
        seed=2021,
        balance_dataset: bool = False,
    ):
        """Convert data into TensorFlow datasets for training and testing.

        Transforms raw data into shuffled and batched TensorFlow datasets. It handles
        both labelled and unlabelled training data, as well as testing data.

        Parameters
        ----------
        - ds_train_labelled: Labelled training data.
        - ds_train_un_labelled: Unlabelled training data.
        - ds_test: Testing data.
        - batch_size (int): Size of the batches for the datasets.
        - buffer (int): Buffer size for shuffling the dataset.
        - seed (int): Random seed for shuffling reproducibility.
        - balance_dataset (bool): If True, balance the training dataset.

        Returns
        -------
        - Labelled and unlabelled training TensorFlow datasets and a testing dataset.
        """
        ds_train_labelled = (
            ds_train_labelled.shuffle(
                buffer_size=buffer, seed=seed, reshuffle_each_iteration=True
            )
            .map(AudioTools.prepare_example, num_parallel_calls=DataBuilder.AUTOTUNE)
            .batch(batch_size=batch_size)
            # .prefetch(DataBuilder.AUTOTUNE)
            if ds_train_labelled
            else None
        )
        ds_test = (
            ds_test.map(
                AudioTools.prepare_test_example, num_parallel_calls=DataBuilder.AUTOTUNE
            )
            .batch(1)
            .prefetch(DataBuilder.AUTOTUNE)
            if ds_test
            else None
        )
        ds_train_un_labelled = (
            ds_train_un_labelled.shuffle(
                buffer_size=buffer, seed=seed + 1, reshuffle_each_iteration=True
            )
            .map(AudioTools.prepare_example, num_parallel_calls=DataBuilder.AUTOTUNE)
            .batch(batch_size=batch_size)
            # .prefetch(DataBuilder.AUTOTUNE)
            if ds_train_un_labelled
            else None
        )

        # make balanced
        if balance_dataset:
            if ds_train_labelled:
                ds_train_labelled = balance_sampler(ds_train_labelled, batch_size)
            # re sampling not required for unlabelled data.
            # if ds_train_un_labelled:
            #     ds_train_un_labelled=balance_sampler(ds_train_un_labelled, batch_size)
        return ds_train_labelled, ds_train_un_labelled, ds_test


def balance_sampler(dataset, batch_size):
    """Address the skew in the speechcommands dataset.

    Here we resample it so we see the same number of instances for all classes. Here we
    use TF's rejection_resample functionality
    https://www.tensorflow.org/guide/data#rejection_resampling.
    """

    # pylint: disable=unused-argument
    def class_func(features, label):
        return label

    # Measure initial distribution over labels
    ddd = np.concatenate([lbl for _, lbl in list(dataset.as_numpy_iterator())])
    initial_dist = np.bincount(ddd) / len(ddd)

    # target distribution is uniform
    target_dist = np.ones_like(initial_dist) / len(initial_dist)

    # resample
    dataset_resampled = (
        dataset.unbatch()
        .rejection_resample(
            class_func, target_dist=target_dist, initial_dist=initial_dist
        )
        .batch(batch_size)
    )

    dataset_balanced = dataset_resampled.map(
        lambda extra_label, features_and_label: features_and_label
    )

    return dataset_balanced
