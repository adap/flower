"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-
processyour dataset (or all of the above). If the desired way of running your baseline
is to first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the
Experiment"block) that this file should be executed first.
"""

import logging
from collections import Counter
from functools import partial
from typing import Any

import medmnist
import torch
import torchvision.transforms as transforms
from datasets import Dataset, DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    PathologicalPartitioner,
    ShardPartitioner,
)
from medmnist import INFO
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from transformers import AutoTokenizer


def _get_medmnist(data_flag="pathmnist", download=True):
    """Load and convert the MedMNIST dataset into a Hugging Face DatasetDict.

    Parameters
    ----------
    data_flag : str, optional
        The key indicating which MedMNIST dataset to load (default is 'pathmnist').
    download : bool, optional
        Whether to download the dataset if not present locally (default is True).

    Returns
    -------
    DatasetDict
        A dictionary with keys "train" and "test", each containing a Hugging Face
        Dataset        object.
    """
    info = INFO[data_flag]
    n_channels = info["n_channels"]
    n_classes = len(info["label"])
    print(f"INFO: {info}, \n  n_channels: {n_channels},  \n n_classes: {n_classes}")

    DataClass = getattr(medmnist, info["python_class"])

    train_dataset = DataClass(split="train", download=download)
    test_dataset = DataClass(split="test", download=download)

    # Convert to Hugging Face Dataset format
    def medmnist_to_hf_dataset(medmnist_dataset):
        data_dict = {"image": [], "label": []}
        for pixels, label in medmnist_dataset:
            data_dict["image"].append(pixels)
            data_dict["label"].append(label.item())
        return Dataset.from_dict(data_dict)

    hf_train_dataset = medmnist_to_hf_dataset(train_dataset)
    hf_test_dataset = medmnist_to_hf_dataset(test_dataset)

    # Combine datasets into a single dataset with splits
    hf_dataset = DatasetDict({"train": hf_train_dataset, "test": hf_test_dataset})

    logging.info("conversion to hf_dataset done")
    return hf_dataset


def tokenize_function_factory(cfg):
    """Create and return a tokenizer function based on the provided configuration.

    The returned function will tokenize examples from the dataset according to the
    dataset type. For "dbpedia_14", it tokenizes the "content" field; for others (like
    Yahoo Answers), it tokenizes the "text" field.

    Parameters
    ----------
    cfg : object
        A configuration object that must include:
            - dname: Name of the dataset.
            - mname: The model name or tokenizer identifier.

    Returns
    -------
    function
        A function that accepts a batch of examples and returns tokenized outputs.
    """
    input_col_name = "content" if cfg.dname == "dbpedia_14" else "text"

    def _default_tokenize_function(examples):
        return tokenizer(
            examples[input_col_name],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    def _yahoo_answers_tokenize_function(examples):
        examples["label"] = examples["topic"]
        return tokenizer(
            examples["question_title"] + " " + examples["question_content"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.mname, trust_remote_code=True)

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if cfg.dname == "yahoo_answers_topics":
        return _yahoo_answers_tokenize_function
    return _default_tokenize_function


def train_test_transforms_factory(cfg):
    """Create and return train and test transformation functions for image datasets.

    Depending on the dataset name specified in the configuration (cfg.dname), this
    function returns a dictionary with keys 'train' and 'test' that map to
    transformation functions.

    Parameters
    ----------
    cfg : object
        A configuration object that must include:
            - dname: Name of the dataset (e.g. "cifar10", "mnist", "pathmnist", etc.).

    Returns
    -------
    dict
        A dictionary with two keys:
            - 'train': Transformation function for training data.
            - 'test': Transformation function for test data.

    Raises
    ------
    ValueError
        If the dataset name is not recognized.
    """
    train_transforms = None
    test_transforms = None
    if cfg.dname == "cifar10":

        def apply_train_transformCifar(example):
            transform = Compose(
                [
                    Resize((32, 32)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )
            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["img"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["img"]
            return example

        def apply_test_transformCifar(example):
            transform = Compose(
                [
                    Resize((32, 32)),
                    ToTensor(),
                    Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]
            )

            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["img"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["img"]
            return example

        train_transforms = apply_train_transformCifar
        test_transforms = apply_test_transformCifar
    elif cfg.dname == "mnist":

        def apply_train_transformMnist(example):

            transform = Compose(
                [Resize((32, 32)), ToTensor(), Normalize((0.1307,), (0.3081,))]
            )
            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["image"]
            ]
            # example['pixel_values'] = transform(example['image'].convert("RGB"))
            example["label"] = torch.tensor(example["label"])
            return example

        def apply_test_transformMnist(example):
            transform = Compose(
                [Resize((32, 32)), ToTensor(), Normalize((0.1307,), (0.3081,))]
            )

            example["pixel_values"] = [
                transform(image.convert("RGB")) for image in example["image"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["image"]

            return example

        train_transforms = apply_train_transformMnist
        test_transforms = apply_test_transformMnist
    elif cfg.dname in ["pathmnist", "organamnist"]:
        tfms = transforms.Compose(
            [Resize((32, 32)), ToTensor(), Normalize(mean=[0.5], std=[0.5])]
        )

        def apply_transform(example):
            example["pixel_values"] = [
                tfms(image.convert("RGB")) for image in example["image"]
            ]
            example["label"] = torch.tensor(example["label"])
            del example["image"]
            return example

        return {"train": apply_transform, "test": apply_transform}

    else:
        raise ValueError(f"Unknown dataset: {cfg.dname}")

    return {"train": train_transforms, "test": test_transforms}


def _initialize_image_dataset(cfg, dat_partitioner_func, fetch_only_test_data):
    """Initialize and process an image dataset by applying train/test transformations.

    This function partitions the dataset using the provided partitioner function,
    then applies the image transformation functions to both client and server datasets.

    Parameters
    ----------
    cfg : object
        A configuration object containing dataset and partitioning parameters.
    dat_partitioner_func : function
        The function that partitions the dataset into client and server splits.
    fetch_only_test_data : bool
        If True, only the test data will be fetched.

    Returns
    -------
    dict
        A dictionary containing:
            - 'client2data': Mapping of client IDs to transformed training data.
            - 'server_data': Transformed test data.
    """
    target_label_col = "label"

    d = dat_partitioner_func(cfg, target_label_col, fetch_only_test_data)
    transforms = train_test_transforms_factory(cfg=cfg)
    d["client2data"] = {
        k: v.map(
            transforms["train"], batched=True, batch_size=256, num_proc=8
        ).with_format("torch")
        for k, v in d["client2data"].items()
    }
    d["server_data"] = (
        d["server_data"]
        .map(transforms["test"], batched=True, batch_size=256, num_proc=8)
        .with_format("torch")
    )
    return d


def _initialize_transformer_dataset(cfg, dat_partitioner_func, fetch_only_test_data):
    """Initialize and process a transformer-based dataset by tokenizing text data.

    The function partitions the dataset using the provided partitioner function and
    applies the appropriate tokenization function to both client and server datasets.

    Parameters
    ----------
    cfg : object
        A configuration object containing dataset and partitioning parameters.
    dat_partitioner_func : function
        The function that partitions the dataset.
    fetch_only_test_data : bool
        If True, only test data is processed.

    Returns
    -------
    dict
        A dictionary with keys 'client2data' and 'server_data' containing tokenized
        data.
    """
    target_label_col = "label"
    if cfg.dname == "yahoo_answers_topics":
        target_label_col = "topic"

    d = dat_partitioner_func(cfg, target_label_col, fetch_only_test_data)
    d["client2data"] = {
        k: v.map(tokenize_function_factory(cfg)) for k, v in d["client2data"].items()
    }
    d["server_data"] = d["server_data"].map(tokenize_function_factory(cfg))
    return d


def _load_dist_based_clients_server_datasets(
    cfg, dat_partitioner_func, fetch_only_test_data: bool = False
):
    """Load and partition datasets.

    Load the requested dataset, split it into client and server parts according to the
    chosen distribution strategy, and return the resulting objects. Both image and
    text (transformer) datasets are supported.

    Parameters
    ----------
    cfg : object
        Configuration object with dataset information (e.g., ``dname``,
        ``architecture``).
    dat_partitioner_func : Callable
        Partitioner used to split the training data across clients.
    fetch_only_test_data : bool, optional
        If ``True``, only the test data is fetched (default is ``False``).

    Returns
    -------
    dict
        Dictionary with the following keys:

        * ``client2data`` – Mapping client‑ID → training split
        * ``server_data`` – Test split for the server
        * ``client2class`` – Per‑client label counts
        * ``fds`` – The ``FederatedDataset`` object used for partitioning

    Raises
    ------
    ValueError
        If the dataset name (``cfg.dname``) or architecture is unknown.
    """
    if cfg.dname in ["cifar10", "mnist", "pathmnist", "organamnist"]:
        return _initialize_image_dataset(
            cfg, dat_partitioner_func, fetch_only_test_data
        )

    if cfg.dname in ["dbpedia_14", "yahoo_answers_topics"]:
        return _initialize_transformer_dataset(
            cfg, dat_partitioner_func, fetch_only_test_data
        )

    if cfg.architecture == "cnn":
        return _initialize_image_dataset(
            cfg, dat_partitioner_func, fetch_only_test_data
        )

    if cfg.architecture == "transformer":
        return _initialize_transformer_dataset(
            cfg, dat_partitioner_func, fetch_only_test_data
        )

    raise ValueError(f"Unknown dataset: {cfg.dname}")


def getLabelsCount(partition, target_label_col):
    """Count the number of occurrences for each label in a dataset partition.

    Parameters
    ----------
    partition : Dataset or list-like
        The dataset partition where each example contains the target label.
    target_label_col : str
        The key corresponding to the label in each example.

    Returns
    -------
    dict
        A dictionary mapping each label to its count.
    """
    label2count = Counter(
        example[target_label_col] for example in partition  # type: ignore
    )  # type: ignore
    return dict(label2count)


def _fix_partition(cfg, c_partition, target_label_col):
    """Clean and truncate a client data partition based on minimum sample requirements.

    This function filters out labels with fewer than 10 occurrences, then limits the
    partition size to a maximum specified by the configuration. It also ensures that
    the final partition size is compatible with the batch size.

    Parameters
    ----------
    cfg : object
        Configuration object with attributes:
            - max_per_client_data_size : int, maximum allowed data per client.
            - batch_size : int, the batch size for training.
    c_partition : Dataset or list-like
        The original dataset partition for a client.
    target_label_col : str
        The key corresponding to the label in each example.

    Returns
    -------
    dict
        A dictionary with keys:
            - 'partition': The cleaned and possibly truncated dataset partition.
            - 'partition_labels_count': The count of labels in the partition.
    """
    label2count = getLabelsCount(c_partition, target_label_col)

    filtered_labels = {
        label: count for label, count in label2count.items() if count >= 10
    }

    indices_to_select = [
        i
        for i, example in enumerate(c_partition)
        if example[target_label_col] in filtered_labels
    ]  # type: ignore

    ds = c_partition.select(indices_to_select)

    assert (
        cfg.max_per_client_data_size > 0
    ), f"max_per_client_data_size: {cfg.max_per_client_data_size}"

    if len(ds) > cfg.max_per_client_data_size:
        # ds = ds.shuffle()
        ds = ds.select(range(cfg.max_per_client_data_size))

    if len(ds) % cfg.batch_size == 1:
        ds = ds.select(range(len(ds) - 1))

    partition_labels_count = getLabelsCount(ds, target_label_col)
    return {"partition": ds, "partition_labels_count": partition_labels_count}


def _partition_helper(
    partitioner, cfg, target_label_col, fetch_only_test_data, subtask
):
    """Partition the dataset among clients and prepare the server test data.

    This helper function uses the provided partitioner to distribute data among clients.
    It then fixes each client partition (if needed) and collects class counts.

    Parameters
    ----------
    partitioner : object
        An instance of a partitioner (e.g. DirichletPartitioner, ShardPartitioner).
    cfg : object
        A configuration object containing:
            - num_clients: Number of clients.
            - max_server_data_size: Maximum number of samples for the server.
            - max_per_client_data_size: Maximum samples allowed per client.
            - batch_size: Batch size used during training.
            - dname: Dataset name.
    target_label_col : str
        The label key used for partitioning.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        If specified, indicates a subset of data to be used.

    Returns
    -------
    dict
        A dictionary with the following keys:
            - 'client2data': Mapping of client IDs to their dataset partitions.
            - 'server_data': The server test dataset.
            - 'client2class': Mapping of client IDs to label counts.
            - 'fds': The FederatedDataset instance used for partitioning.
    """
    # logging.info(f"Dataset name: {cfg.dname}")
    clients_class = []
    clients_data = []
    server_data = None
    fds = None
    if cfg.dname in ["pathmnist", "organamnist"]:
        hf_dataset = _get_medmnist(data_flag=cfg.dname, download=True)

        partitioner.dataset = hf_dataset["train"]
        fds = partitioner

        logging.info(f"max data size {cfg.max_server_data_size}")

        if cfg.max_server_data_size < len(hf_dataset["test"]):
            server_data = hf_dataset["test"].select(range(cfg.max_server_data_size))
        else:
            server_data = hf_dataset["test"]

    # partition = partitioner.load_partition(partition_id=partition_id)
    # return partition
    else:
        if subtask is not None:
            fds = FederatedDataset(
                dataset=cfg.dname, partitioners={"train": partitioner}, subset=subtask
            )
        else:
            fds = FederatedDataset(
                dataset=cfg.dname, partitioners={"train": partitioner}
            )

        if len(fds.load_split("test")) < cfg.max_server_data_size:
            server_data = fds.load_split("test")
        else:
            server_data = fds.load_split("test").select(range(cfg.max_server_data_size))
    logging.info(f"Partition helper: Keys in the dataset are: {server_data[0].keys()}")

    for cid in range(cfg.num_clients):
        client_partition = fds.load_partition(cid)
        temp_dict = {}

        if cfg.max_per_client_data_size > 0:
            logging.info(f" Fixing partition for client {cid}")
            temp_dict = _fix_partition(cfg, client_partition, target_label_col)
        else:
            logging.info(f" No data partition fix requried for client {cid}")
            temp_dict = {
                "partition": client_partition,
                "partition_labels_count": getLabelsCount(
                    client_partition, target_label_col
                ),
            }

        if len(temp_dict["partition"]) >= cfg.batch_size:
            clients_data.append(temp_dict["partition"])
            clients_class.append(temp_dict["partition_labels_count"])

    logging.info(" -- fix partition is done --")
    client2data = {f"{id}": v for id, v in enumerate(clients_data)}
    client2class = {f"{id}": v for id, v in enumerate(clients_class)}
    return {
        "client2data": client2data,
        "server_data": server_data,
        "client2class": client2class,
        "fds": fds,
    }


def _dirichlet_data_distribution(
    cfg, target_label_col, fetch_only_test_data, subtask=None
):
    """Partition the dataset among clients using a Dirichlet distribution.

    Parameters
    ----------
    cfg : object
        Configuration object containing:
            - num_clients: Number of clients.
            - dirichlet_alpha: Alpha parameter for the Dirichlet distribution.
    target_label_col : str
        Key used for partitioning by label.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        A subset indicator for partitioning if needed.

    Returns
    -------
    dict
        A dictionary with keys 'client2data', 'server_data', 'client2class', and 'fds'.
    """
    partitioner = DirichletPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        alpha=cfg.dirichlet_alpha,
        min_partition_size=0,
        self_balancing=True,
        shuffle=True,
    )

    return _partition_helper(
        partitioner, cfg, target_label_col, fetch_only_test_data, subtask
    )


def _sharded_data_distribution(
    num_classes_per_partition, cfg, target_label_col, fetch_only_test_data, subtask=None
):
    """Partition the dataset among clients using a sharded non-IID distribution.

    This function uses a shard partitioner that assigns a fixed number of classes
    per client.

    Parameters
    ----------
    num_classes_per_partition : int
        Number of classes that each client should receive.
    cfg : object
        Configuration object containing:
            - num_clients: Number of clients.
    target_label_col : str
        The key used for partitioning by label.
    fetch_only_test_data : bool
        If True, only test data is processed.
    subtask : optional
        A subset indicator for partitioning if needed.

    Returns
    -------
    dict
        A dictionary with keys 'client2data', 'server_data', 'client2class', and 'fds'.
    """
    partitioner = ShardPartitioner(
        num_partitions=cfg.num_clients,
        partition_by=target_label_col,
        shard_size=2000,
        num_shards_per_partition=num_classes_per_partition,
        shuffle=True,
    )
    return _partition_helper(
        partitioner, cfg, target_label_col, fetch_only_test_data, subtask
    )


def _pathological_partitioner(
    dataset: Dataset, num_clients: int, alpha: float = 0.5, cfg: Any = None
) -> DatasetDict:
    """Partition the dataset among clients using a pathological strategy.

    This function creates a non-IID partition of the dataset by sorting examples by
    label and then distributing them in a way that creates highly skewed
    distributions per client.

    Parameters
    ----------
    dataset : Dataset
        The dataset to partition.
    num_clients : int
        Number of clients to partition the data among.
    alpha : float, optional
        Concentration parameter for the Dirichlet distribution (default is 0.5).
    cfg : Any, optional
        Configuration object containing dataset parameters.

    Returns
    -------
    DatasetDict
        A dictionary mapping client IDs to their respective datasets.
    """
    partitioner = PathologicalPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        num_classes_per_partition=1,
        shuffle=True,
        class_assignment_mode="deterministic",
    )
    return _partition_helper(partitioner, cfg, "label", False, None)


class ClientsAndServerDatasets:
    """Class for managing client and server datasets.

    This class handles the organization and management of datasets for both clients and
    the server in the federated learning system.
    """

    def __init__(self, cfg):
        """Initialize the dataset manager.

        Args:
            cfg: Configuration object containing dataset parameters
        """
        self.cfg = cfg
        self.data_dist_partitioner_func = None
        self._set_distriubtion_partitioner()
        self._setup()

    def _set_distriubtion_partitioner(self):
        """Set the data distribution partitioner function based on the configuration.

        The method selects the partitioner function to use (e.g. Dirichlet, sharded, or
        pathological) based on cfg.data_dist.dist_type.
        """
        if self.cfg.tool.tracefl.data_dist.dist_type == "non_iid_dirichlet":
            self.data_dist_partitioner_func = _dirichlet_data_distribution
        elif self.cfg.tool.tracefl.data_dist.dist_type == "sharded-non-iid-1":
            self.data_dist_partitioner_func = partial(
                _sharded_data_distribution, 1
            )  # passing num_classes_per_partition
        elif self.cfg.tool.tracefl.data_dist.dist_type == "sharded-non-iid-2":
            self.data_dist_partitioner_func = partial(_sharded_data_distribution, 2)
        elif self.cfg.tool.tracefl.data_dist.dist_type == "sharded-non-iid-3":
            self.data_dist_partitioner_func = partial(_sharded_data_distribution, 3)
        elif self.cfg.tool.tracefl.data_dist.dist_type == "PathologicalPartitioner-1":
            self.data_dist_partitioner_func = partial(_pathological_partitioner, 1)
        elif self.cfg.tool.tracefl.data_dist.dist_type == "PathologicalPartitioner-2":
            self.data_dist_partitioner_func = partial(_pathological_partitioner, 2)
        elif self.cfg.tool.tracefl.data_dist.dist_type == "PathologicalPartitioner-3":
            self.data_dist_partitioner_func = partial(_pathological_partitioner, 3)
        else:
            raise ValueError(
                f"Unknown distribution type: {self.cfg.tool.tracefl.data_dist.dist}"
            )

    def _setup_hugging_dataset(self):
        """Set up Hugging Face dataset for federated learning.

        This function configures and prepares a Hugging Face dataset for use
        in the federated learning system.

        Args:
            cfg: Configuration object containing dataset parameters

        Returns
        -------
            Configured Hugging Face dataset
        """
        d = _load_dist_based_clients_server_datasets(
            self.cfg.tool.tracefl.data_dist, self.data_dist_partitioner_func
        )
        self.client2data = d["client2data"]

        self.server_testdata = d["server_data"]
        self.client2class = d["client2class"]
        self.fds = d["fds"]
        logging.info(f"client2class: {self.client2class}")

        logging.info(f"> client2class {self.client2class}")

        data_per_client = [len(dl) for dl in self.client2data.values()]
        logging.info(f"Data per client in experiment {data_per_client}")
        min_data = min(len(dl) for dl in self.client2data.values())
        logging.info(f"Min data on a client: {min_data}")

    def _setup(self):
        """Initialize the Hugging Face dataset.

        This method sets up the Hugging Face dataset for federated learning.
        """
        self._setup_hugging_dataset()

    def get_data(self):
        """Retrieve the prepared client and server datasets for federated simulation.

        Returns
        -------
        dict
            A dictionary containing:
                - 'server_testdata': The server's test dataset.
                - 'client2class': Label count per client.
                - 'client2data': Client training datasets.
                - 'fds': The FederatedDataset object used for partitioning.
        """
        return {
            "server_testdata": self.server_testdata,
            "client2class": self.client2class,
            "client2data": self.client2data,
            "fds": self.fds,
        }


def _initialize_dataset(cfg_dataset):
    """Initialize the dataset based on configuration.

    Parameters
    ----------
    cfg_dataset : object
        Configuration object with dataset information
        and parameters for initialization.

    Returns
    -------
    Dataset
        The initialized dataset.
    """
    import torchvision  # Import here to avoid circular imports

    if cfg_dataset.name == "cifar10":
        return _initialize_image_dataset(
            cfg_dataset,
            torchvision.datasets.CIFAR10,
            "CIFAR10",
        )
    elif cfg_dataset.name == "cifar100":
        return _initialize_image_dataset(
            cfg_dataset,
            torchvision.datasets.CIFAR100,
            "CIFAR100",
        )
