"""Required imports for utils.py script."""

import collections
import itertools
import random

import numpy as np
import tensorflow as tf


class AudioTools:  # pylint: disable=too-many-instance-attributes
    """Provide static methods for audio data processing in a machine learning context.

    This class includes methods for reading, padding, extracting, and transforming audio
    waveforms into features suitable for machine learning models.
    """

    @staticmethod
    def read_audio(path, label):
        """Read an audio file and return its waveform along with the label.

        Parameters
        ----------
        - path: The file path of the audio file.
        - label: The associated label of the audio file.

        Returns
        -------
        - Tuple containing the waveform of the audio file and its label.
        """
        raw_audio = tf.io.read_file(path)
        waveform, _ = tf.audio.decode_wav(raw_audio)
        waveform = waveform[Ellipsis, 0]
        return waveform, label

    @staticmethod
    def pad(waveform, sequence_length=16000):
        """Pad the given waveform to a specified sequence length.

        Parameters
        ----------
        - waveform: The waveform to be padded.
        - sequence_length: The length to which the waveform should be padded.

        Returns
        -------
        - The padded waveform.
        """
        padding = tf.maximum(sequence_length - tf.shape(waveform)[0], 0)
        left_pad = padding // 2
        right_pad = padding - left_pad
        return tf.pad(waveform, paddings=[[left_pad, right_pad]])

    @staticmethod
    def extract_window(waveform, seg_length=15690):
        """Extract a window of specified length from the given waveform.

        Parameters
        ----------
        - waveform: The input waveform.
        - seg_length: The length of the segment to extract.

        Returns
        -------
        - A segment of the waveform of the specified length.
        """
        waveform = AudioTools.pad(waveform)
        return tf.image.random_crop(waveform, [seg_length])

    # pylint: disable=too-many-arguments, too-many-locals
    @staticmethod
    def extract_spectrogram(
        waveform,
        sample_rate=16000,
        frame_length=400,
        frame_step=160,
        fft_length=1024,
        n_mels=64,
        fmin=60.0,
        fmax=7800.0,
    ):
        """Convert a waveform to a log-mel spectrogram.

        Parameters
        ----------
        - waveform: The input waveform.
        - sample_rate: Sampling rate of the audio.
        - frame_length: Length of each audio frame.
        - frame_step: Step size between frames.
        - fft_length: Length of the Fast Fourier Transform.
        - n_mels: Number of Mel bands.
        - fmin: Minimum frequency.
        - fmax: Maximum frequency.

        Returns
        -------
        - Log-mel spectrogram of the waveform.
        """
        stfts = tf.signal.stft(
            waveform,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
        )
        spectrograms = tf.abs(stfts)
        num_spectrogram_bins = stfts.shape[-1]
        lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins,
            num_spectrogram_bins,
            sample_rate,
            lower_edge_hertz,
            upper_edge_hertz,
        )
        mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
        mel_spectrograms.set_shape(
            spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:])
        )
        mel_spectrograms = tf.clip_by_value(
            mel_spectrograms, clip_value_min=1e-5, clip_value_max=1e8
        )
        log_mel_spectrograms = tf.math.log(mel_spectrograms)
        return log_mel_spectrograms[Ellipsis, tf.newaxis]

    @staticmethod
    def prepare_example(waveform, label):
        """Prepare an audio example for model training or evaluation.

        Parameters
        ----------
        - waveform: The input waveform.
        - label: The label of the audio example.

        Returns
        -------
        - Prepared log-mel spectrogram feature and the corresponding label.
        """
        waveform = AudioTools.pad(waveform)
        waveform = tf.math.l2_normalize(waveform, epsilon=1e-9)
        waveform = AudioTools.extract_window(waveform)
        log_mel_spectrogram = AudioTools.extract_spectrogram(waveform)
        return log_mel_spectrogram, label

    @staticmethod
    def prepare_test_example(waveform, label):
        """Prepare an audio example specifically for testing.

        Parameters
        ----------
        - waveform: The input waveform.
        - label: The label of the audio example.

        Returns
        -------
        - Prepared log-mel spectrogram feature for testing and the corresponding label.
        """
        waveform = AudioTools.pad(waveform)
        waveform = tf.signal.frame(
            waveform, frame_length=98 * 160, frame_step=98 * 160, pad_end=False
        )
        waveform = tf.math.l2_normalize(waveform, axis=-1, epsilon=1e-9)
        log_mel_spectrogram = AudioTools.extract_spectrogram(waveform)
        return log_mel_spectrogram, label


class DataTools:  # pylint: disable=too-many-instance-attributes
    """Provide methods for dataset manipulation and distribution in federated learning.

    Includes methods for calculating statistics, distributing datasets among clients,
    and transforming datasets for semi-supervised learning.
    """

    @staticmethod
    def get_num_samples(dataset):
        """Get the number of samples in a dataset.

        Parameters
        ----------
        - dataset: The dataset to evaluate.

        Returns
        -------
        - The number of samples in the dataset.
        """
        return len(dataset[0])

    @staticmethod
    def get_statistics(num_samples, num_clients, var):
        """Calculate statistical parameters for dataset distribution.

        Parameters
        ----------
        - num_samples: Total number of samples in the dataset.
        - num_clients: Number of clients in federated learning.
        - var: Variance factor for distribution.

        Returns
        -------
        - Calculated mean, variance, limits, and error values for distribution.
        """
        mean = num_samples / num_clients
        var = int(var * mean)
        minimum, maximum = mean - var, int(mean + var)
        error = num_samples - minimum * num_clients  # values to be distributed
        round_error = (error - int(error)) + ((mean - int(mean)) * num_clients)
        return mean, var, (int(minimum), maximum), (int(error), round_error)

    @staticmethod
    def get_class_distribution(dataset):
        """Determine the class distribution in a dataset.

        Parameters
        ----------
        - dataset: The dataset whose class distribution is to be analyzed.

        Returns
        -------
        - A counter object representing the class distribution.
        """
        return collections.Counter(dataset[1])

    @staticmethod
    def split_class_samples(dataset, partitions=1):
        """Split class samples in a dataset into specified number of partitions.

        Parameters
        ----------
        - dataset: The dataset to be split.
        - partitions: The number of partitions.

        Returns
        -------
        - A list of dataset partitions.
        """
        total = len(dataset[0])
        size = total // partitions
        rest = total % partitions
        ranges = []
        assert size != 0 and partitions > 0
        if rest:
            index = list(range(0, total, size))
            extra = [index[i] + i for i in range(rest + 1)] + [
                x + rest for x in index[rest + 1 :][: partitions - rest]
            ]
            ranges = [(extra[i], extra[i + 1]) for i in range(len(extra) - 1)]
        else:
            index = list(range(0, total + 1, size))
            ranges = [(index[i], index[i + 1]) for i in range(len(index) - 1)]
        return [(dataset[0][i:j], dataset[1][i:j]) for i, j in ranges]

    @staticmethod
    def get_dataset_class_samples(dataset, class_number):
        """Extract and return samples of a specific class from the dataset.

        Parameters
        ----------
        - dataset: The dataset from which to extract samples.
        - class_number: The class number to filter by.

        Returns
        -------
        - Tuple of lists: (samples, labels) of the specified class.
        """
        dataset = list(
            zip(
                *[
                    (dataset[0][idx], label)
                    for idx, label in enumerate(dataset[1])
                    if label == class_number
                ]
            )
        )
        return list(dataset[0]), list(dataset[1])

    @staticmethod
    def get_subset(dataset, percentage, num_classes, u_per=1.0, seed=2021):
        """Generate labeled and unlabeled subsets based on the given percentage.

        Parameters
        ----------
        - dataset: The dataset to split into subsets.
        - percentage: The percentage of data to include in the labeled subset.
        - num_classes: The number of classes in the dataset.
        - u_per: The percentage of data to include in the unlabeled subset.
        - seed: The seed for random operations to ensure reproducibility.

        Returns
        -------
        - Two tuples: The labeled subset and the unlabeled subset.
        """
        # num_samples = DataTools.get_num_samples(dataset=dataset)
        class_distribution = DataTools.get_class_distribution(dataset=dataset)
        class_distribution = dict(sorted(class_distribution.items()))
        class_samples = {k: int(percentage * v) for k, v in class_distribution.items()}
        class_samples = dict(sorted(class_samples.items()))
        dataset = list(zip(dataset[0], dataset[1]))
        dataset_per_class = [
            [s for s in dataset if s[1] == i] for i in range(num_classes)
        ]
        class_num_samples = list(class_samples.values())
        # Get Subset of labelled samples
        subset = list(
            itertools.chain(
                *[
                    [
                        dataset_per_class[i].pop(
                            random.Random(seed).randrange(len(dataset_per_class[i]))
                        )
                        for _ in range(class_num_samples[i])
                    ]
                    for i in range(num_classes)
                ]
            )
        )
        subset_u = list(itertools.chain(*dataset_per_class))
        random.Random(seed).shuffle(subset)
        random.Random(seed).shuffle(subset_u)
        subset_u = (
            subset_u[0 : int((len(subset_u) - 1) * u_per)] if u_per < 1.0 else subset_u
        )
        return (subset, len(subset)), (subset_u, len(subset_u))

    @staticmethod
    def distribute_per_samples(dataset, num_clients, variance=0.25, seed=2021):
        """Distribute dataset samples evenly among a specified number of clients.

        Parameters
        ----------
        - dataset: The dataset to be distributed.
        - num_clients: The number of clients among whom to distribute the dataset.
        - variance: The variance allowed in the distribution of samples.
        - seed: The seed for random operations to ensure reproducibility.

        Returns
        -------
        - Generator yielding dataset subsets for each client.
        """
        dataset = tuple(list(t) for t in zip(*dataset))
        num_samples = DataTools.get_num_samples(dataset=dataset)
        _, _, limits, errors = DataTools.get_statistics(
            num_samples=num_samples, num_clients=num_clients, var=variance
        )
        distribution = DataTools.create_distribution(
            num_clients=num_clients,
            num_samples=num_samples,
            remain_samples=errors[0],
            minimum=limits[0],
            maximum=limits[1],
            round_error=errors[1],
            seed=seed,
        )
        samples, labels = dataset
        dataset = DataTools.shuffle_dataset(dataset=dataset, seed=seed)
        iter_samples, iter_labels = iter(samples), iter(labels)
        for i in range(num_clients):
            yield list(itertools.islice(iter_samples, distribution[i])), list(
                itertools.islice(iter_labels, distribution[i])
            )

    # pylint: disable=too-many-locals
    @staticmethod
    def distribute_per_class(
        dataset, num_clients, num_classes, class_variance=0.0, seed=2021
    ):
        """Distribute dataset based on class distribution among clients.

        Parameters
        ----------
        - dataset: The dataset to be distributed.
        - num_clients: The number of clients.
        - num_classes: The number of classes in the dataset.
        - class_variance: The variance in the number of classes per client.
        - seed: The seed for random operations to ensure reproducibility.

        Returns
        -------
        - A list of datasets for each client, distributed based on class.
        """
        dataset = tuple(list(t) for t in zip(*dataset))
        (overlap, _num_classes_, _num_clients_) = (
            (True, num_clients, num_classes)
            if num_classes < num_clients
            else (False, num_classes, num_clients)
        )
        _, _, limits, errors = DataTools.get_statistics(
            num_samples=_num_classes_, num_clients=_num_clients_, var=class_variance
        )
        distribution = DataTools.create_distribution(
            num_clients=_num_clients_,
            num_samples=_num_classes_,
            remain_samples=errors[0],
            minimum=limits[0],
            maximum=limits[1],
            round_error=errors[1],
            seed=seed,
        )

        if (
            overlap
        ):  # Clients will contain only one class and clients classes will overlap
            distribution = list(
                itertools.chain(
                    *[[idx] * clients for idx, clients in enumerate(distribution)]
                )
            )
            random.Random(seed).shuffle(distribution)
        else:
            classes = list(range(_num_classes_))
            random.Random(seed).shuffle(classes)
            _iter_ = iter(classes)

            def distribute_fun():
                return (
                    yield from (
                        list(itertools.islice(_iter_, distribution[i]))
                        for i in range(_num_clients_)
                    )
                )

            distribution = list(distribute_fun())
        class_datasets = [
            DataTools.get_dataset_class_samples(
                dataset=dataset, class_number=class_number
            )
            for class_number in range(num_classes)
        ]

        # Distribute dataset to clients according to distribution
        if overlap:
            dataset_use = dict(sorted(collections.Counter(distribution).items()))
            class_datasets = [
                (
                    class_datasets[c]
                    if dataset_use[c] == 1
                    else DataTools.split_class_samples(
                        class_datasets[c], dataset_use[c]
                    )
                )
                for c in range(num_classes)
            ]
            parts = list(
                itertools.chain.from_iterable(
                    [list(range(len(c))) for c in class_datasets if isinstance(c, list)]
                )
            )
            _iter_ = iter(parts)
            datasets = [
                (
                    class_datasets[i]
                    if dataset_use[i] == 1
                    else class_datasets[i][next(_iter_)]
                )
                for i in distribution
            ]
            datasets = [DataTools.shuffle_dataset(dataset) for dataset in datasets]
        else:
            datasets = [
                [class_datasets[class_number] for class_number in i]
                for i in distribution
            ]
            datasets = [np.hstack((datasets[i])).tolist() for i in range(len(datasets))]
            datasets = [tuple(data) for data in datasets]
            datasets = [DataTools.shuffle_dataset(dataset) for dataset in datasets]

        datasets = [
            [dataset[0], [int(label) for label in dataset[1]]] for dataset in datasets
        ]
        return datasets

    @staticmethod
    def distribute_per_speaker(dataset, num_speakers, seed=2021):
        """Distribute the dataset based on speaker identity.

        Parameters
        ----------
        - dataset: The dataset to be distributed.
        - num_speakers: The number of distinct speakers in the dataset.
        - seed: The seed for random operations to ensure reproducibility.

        Returns
        -------
        - A list of datasets, each corresponding to a different speaker.
        """
        random.seed(seed)
        np.random.seed(seed)
        dataset = tuple(list(t) for t in zip(*dataset))
        datasets = [
            [
                (file, label)
                for file, label in zip(dataset[0], dataset[1])
                if int(((file.split("/")[-1]).split(".")[0]).split("-")[-1]) == i + 1
            ]
            for i in range(num_speakers)
        ]
        datasets = [list(zip(*dataset)) for dataset in datasets]
        datasets = [(list(dataset[0]), list(dataset[1])) for dataset in datasets]
        return datasets

    # pylint: disable=too-many-arguments
    @staticmethod
    def distribute_per_class_with_class_limit(
        dataset,
        num_clients,
        num_classes,
        mean_class_distribution=3,
        class_variance=0.0,
        seed=2021,
    ):
        """Distribute the dataset with a limit on the number of classes per client.

        Parameters
        ----------
        - dataset: The dataset to be distributed.
        - num_clients: The number of clients.
        - num_classes: The number of classes in the dataset.
        - mean_class_distribution: The mean number of classes per client.
        - class_variance: The variance in the number of classes per client.
        - seed: The seed for random operations to ensure reproducibility.

        Returns
        -------
        - A list of datasets for each client, with a controlled class distribution.
        """
        dataset = tuple(list(t) for t in zip(*dataset))
        random.seed(seed)
        np.random.seed(seed)
        probability_distribution = np.random.normal(
            mean_class_distribution, np.sqrt(class_variance), num_clients
        )
        probability_distribution = [int(p) for p in probability_distribution]
        # Create class distribution across clients
        distributions = [
            list(np.random.choice(a=num_classes, size=p, replace=False))
            for p in probability_distribution
        ]
        # Find number of clients that share dataset.
        dataset_use = dict(
            sorted(collections.Counter(list(itertools.chain(*distributions))).items())
        )
        # Make sure that all classes are used. Otherwise change seed.
        assert (
            len(dataset_use) == num_classes
        ), "Not all classes were selected. Choose a different seed!"
        # Split classes into clients partitions
        class_datasets = [
            DataTools.get_dataset_class_samples(
                dataset=dataset, class_number=class_number
            )
            for class_number in range(num_classes)
        ]
        class_datasets = [
            (
                class_datasets[c]
                if dataset_use[c] == 1
                else DataTools.split_class_samples(class_datasets[c], dataset_use[c])
            )
            for c in range(num_classes)
        ]
        # Keep track of splitted partitions
        parts = [list(np.arange(i)) for i in list(dataset_use.values())]
        # Fill datasets with examples from class distribution.
        datasets = [
            [
                (
                    class_datasets[j]
                    if (len(parts[j]) == 1 and isinstance(class_datasets[j], tuple))
                    else class_datasets[j][parts[j].pop()]
                )
                for j in dis
            ]
            for dis in distributions
        ]
        # Flatten clients datasets to ([samples],[labels]) format.
        datasets = [
            [
                list(itertools.chain(*list(zip(*d))[0])),
                list(itertools.chain(*list(zip(*d))[1])),
            ]
            for d in datasets
        ]
        # Shuffle each dataset.
        datasets = [DataTools.shuffle_dataset(d) for d in datasets]
        return datasets

    # pylint: disable=too-many-arguments
    @staticmethod
    def create_distribution(
        num_clients,
        num_samples,
        remain_samples,
        minimum,
        maximum,
        round_error,
        seed=2021,
    ):
        """Create a distribution plan for distributing samples among clients.

        Parameters
        ----------
        - num_clients: The number of clients.
        - num_samples: The total number of samples to distribute.
        - remain_samples: Remaining samples to distribute after initial division.
        - minimum: The minimum number of samples per client.
        - maximum: The maximum number of samples per client.
        - round_error: The error due to rounding in distribution.
        - seed: The seed for random operations to ensure reproducibility.

        Returns
        -------
        - A list representing the distribution of samples among clients.
        """
        distribution = [minimum] * num_clients
        random.seed(seed)
        while remain_samples > 0:
            idx = random.randint(0, len(distribution) - 1)
            distribution[idx], remain_samples = (
                (distribution[idx] + 1, remain_samples - 1)
                if distribution[idx] < maximum
                else (distribution[idx], remain_samples)
            )
        # Add error due to rounding to random positions.
        for _i in range(round(round_error)):
            idx = random.randint(0, len(distribution) - 1)
            distribution[idx] = (
                distribution[idx] + 1
                if sum(distribution) + 1 <= num_samples
                else distribution[idx]
            )
        return distribution

    @staticmethod
    def convert_to_unlabelled(dataset, unlabelled_data_identifier=-1):
        """Convert a labeled dataset to an unlabeled one by replacing labels.

        Replaces the labels in the dataset with a specified identifier, effectively
        converting it to an unlabeled dataset. This is useful in semi-supervised
        learning scenarios.

        Parameters
        ----------
        - dataset: The dataset to be converted, consisting of data and labels.
        - unlabelled_data_identifier: The identifier to replace the labels with,
        defaulting to -1.

        Returns
        -------
        - A list of tuples, where each tuple contains data and unlabelled identifier.
        """
        return [(data[0], unlabelled_data_identifier) for data in dataset]

    @staticmethod
    def shuffle_dataset(dataset, seed=2021):
        """Shuffle a dataset in a reproducible manner.

        Randomly shuffles the dataset using a specified seed to ensure the shuffle
        can be reproduced.

        Parameters
        ----------
        - dataset: The dataset to shuffle, consisting of data and labels.
        - seed: The seed for the random number generator to ensure reproducibility,
        defaulting to 2021.

        Returns
        -------
        - The shuffled dataset as a tuple of two lists, one for data and one for labels.
        """
        random.seed(seed)
        dataset = list(zip(dataset[0], dataset[1]))
        random.shuffle(dataset)
        return tuple(list(t) for t in zip(*dataset))
