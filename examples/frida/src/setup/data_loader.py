import pickle

import numpy as np
import torch
import torchvision.transforms as transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import datasets


def create_apply_transforms(cfg):
    def apply_transforms(batch):
        if cfg.dataset == "fmnist":
            transform = transforms.Compose(
                [
                    transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

        elif cfg.dataset == "cifar10":
            transform = transforms.Compose(
                [
                    transforms.Resize(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )

        else:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(
                        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
                    ),
                ]
            )

        batch[cfg.image_name] = [transform(img) for img in batch[cfg.image_name]]
        return batch

    return apply_transforms


def load_datasets_offline(cfg):
    if cfg.iid:
        partitioner = IidPartitioner(cfg.num_clients)
    else:
        partitioner = DirichletPartitioner(
            cfg.num_clients,
            partition_by=cfg.image_label,
            alpha=cfg.dirichlet_alpha,
            min_partition_size=0,
        )

    train_x, train_y = load_data("{}train_data.pkl".format(cfg.path_to_local_dataset))
    test_x, test_y = load_data("{}test_data.pkl".format(cfg.path_to_local_dataset))

    train_pil_images = images_to_pil(train_x, cfg.dataset)
    print("train pil images", train_pil_images)
    test_pil_images = images_to_pil(test_x, cfg.dataset)

    train_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in train_y]
    test_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in test_y]
    partitioner.dataset = datasets.Dataset.from_dict(
        {cfg.image_name: train_pil_images, cfg.image_label: train_y}
    )

    apply_transforms = create_apply_transforms(cfg)
    trainloaders = []
    valloaders = []
    for partition_id in range(cfg.num_clients):
        partition_client = partitioner.load_partition(partition_id)
        partition_client = partition_client.with_transform(apply_transforms)
        partition_client = partition_client.train_test_split(train_size=0.8)
        train_samples = min(cfg.train_samples, len(partition_client["train"]))
        partition_train = partition_client["train"].select(range(train_samples))
        test_samples = min(cfg.test_samples, len(partition_client["test"]))
        partition_val = partition_client["test"].select(range(test_samples))
        print(
            "Client {}: {} train samples, {} test samples".format(
                partition_id, len(partition_train), len(partition_val)
            )
        )

        trainloaders.append(
            DataLoader(partition_train, batch_size=cfg.batch_size, shuffle=True)
        )
        valloaders.append(
            DataLoader(partition_val, batch_size=cfg.batch_size, shuffle=True)
        )

    testset = datasets.Dataset.from_dict(
        {cfg.image_name: test_pil_images, cfg.image_label: test_y}
    ).with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=cfg.batch_size, shuffle=True)

    return trainloaders, valloaders, testloader


def load_datasets(cfg):
    if cfg.iid:
        partitioner = IidPartitioner(cfg.num_clients)
    else:
        partitioner = DirichletPartitioner(
            cfg.num_clients,
            partition_by=cfg.image_label,
            alpha=cfg.dirichlet_alpha,
            min_partition_size=0,
        )

    fds = FederatedDataset(dataset=cfg.dataset, partitioners={"train": partitioner})

    apply_transforms = create_apply_transforms(cfg)
    trainloaders = []
    valloaders = []
    for partition_id in range(cfg.num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        trainloaders.append(DataLoader(partition["train"], batch_size=cfg.batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=cfg.batch_size))

    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=cfg.batch_size)

    return trainloaders, valloaders, testloader


def load_subsets(cfg):
    num_clients = cfg.num_clients
    num_rounds = cfg.num_rounds
    dataset = cfg.dataset
    batch_size = cfg.batch_size
    if cfg.iid:
        partitioner = IidPartitioner(num_clients)
    else:
        partitioner = DirichletPartitioner(
            num_clients,
            partition_by=cfg.image_label,
            alpha=cfg.dirichlet_alpha,
            min_partition_size=0,
        )

    fds = FederatedDataset(dataset=dataset, partitioners={"train": partitioner})

    apply_transforms = create_apply_transforms(cfg)
    trainloaders = []
    valloaders = []
    subsetloader = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8)
        train_samples = min(cfg.train_samples, len(partition["train"]))
        partition_train = partition["train"].select(range(train_samples))
        test_samples = min(cfg.test_samples, len(partition["test"]))
        partition_test = partition["test"].select(range(test_samples))
        print(
            "Client {}: {} train samples, {} test samples".format(
                partition_id, len(partition_train), len(partition_test)
            )
        )
        trainloaders.append(
            DataLoader(partition_train, batch_size=batch_size, shuffle=True)
        )
        valloaders.append(
            DataLoader(partition_test, batch_size=batch_size, shuffle=True)
        )

    subset = fds.load_split("test").with_transform(apply_transforms)
    subset_samples = min(cfg.subset_samples, len(subset))
    max_rounds = len(subset) / num_rounds

    if cfg.dynamic_canary:
        subsetloader = []
        for x in range(num_rounds):
            if x >= max_rounds:
                x = 0
            subsetloader.append(
                DataLoader(
                    subset.select(
                        range(x * subset_samples, x * subset_samples + subset_samples)
                    ),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )

    else:
        subsetloader = DataLoader(
            subset.select(range(subset_samples)), batch_size=batch_size, shuffle=True
        )

    if cfg.noise:
        noise_dataset = RandomNoiseDataset(subset_samples, (3, 32, 32), cfg)
        subsetloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    return trainloaders, valloaders, subsetloader


def images_to_pil(x, dataset_name: str):
    out = []

    for idx, img in enumerate(x):
        if isinstance(img, list):
            arr = np.array(img)
        elif isinstance(img, torch.Tensor):
            arr = img.numpy()
        else:
            arr = np.asarray(img)

        if dataset_name == "fmnist":
            if idx == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(28, 28, 1)
            elif (
                arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3)
            ):
                arr = np.transpose(arr, (1, 2, 0))

            if arr.shape[:2] != (28, 28):
                arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((28, 28)))

        else:
            if arr.ndim == 1:
                arr = arr.reshape(32, 32, 3)
            elif (
                arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3)
            ):
                arr = np.transpose(arr, (1, 2, 0))
            elif arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)

            if arr.shape[:2] != (32, 32):
                arr = np.array(Image.fromarray(arr.astype(np.uint8)).resize((32, 32)))

            out.append(Image.fromarray(arr.astype(np.uint8)))

    return out


def load_subsets_offline(cfg):
    num_clients = cfg.num_clients
    num_rounds = cfg.num_rounds
    batch_size = cfg.batch_size
    if cfg.iid:
        partitioner = IidPartitioner(num_clients)
    else:
        partitioner = DirichletPartitioner(
            num_clients,
            partition_by=cfg.image_label,
            alpha=cfg.dirichlet_alpha,
            min_partition_size=0,
        )

    train_x, train_y = load_data("{}train_data.pkl".format(cfg.path_to_local_dataset))
    test_x, test_y = load_data("{}test_data.pkl".format(cfg.path_to_local_dataset))

    train_pil_images = images_to_pil(train_x, cfg.dataset)
    test_pil_images = images_to_pil(test_x, cfg.dataset)

    train_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in train_y]
    test_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in test_y]
    partitioner.dataset = datasets.Dataset.from_dict(
        {cfg.image_name: train_pil_images, cfg.image_label: train_y}
    )

    apply_transforms = create_apply_transforms(cfg)
    trainloaders = []
    valloaders = []
    subsetloader = []
    for partition_id in range(num_clients):
        partition_client = partitioner.load_partition(partition_id)
        partition_client = partition_client.with_transform(apply_transforms)
        partition_client = partition_client.train_test_split(train_size=0.8)
        train_samples = min(cfg.train_samples, len(partition_client["train"]))
        partition_train = partition_client["train"].select(range(train_samples))
        test_samples = min(cfg.test_samples, len(partition_client["test"]))
        partition_val = partition_client["test"].select(range(test_samples))
        print(
            "Client {}: {} train samples, {} test samples".format(
                partition_id, len(partition_train), len(partition_val)
            )
        )

        trainloaders.append(
            DataLoader(partition_train, batch_size=batch_size, shuffle=True)
        )
        valloaders.append(
            DataLoader(partition_val, batch_size=batch_size, shuffle=True)
        )

    subset = datasets.Dataset.from_dict(
        {cfg.image_name: test_pil_images, cfg.image_label: test_y}
    ).with_transform(apply_transforms)
    subset_samples = min(cfg.subset_samples, len(subset))

    if cfg.dynamic_canary:
        subsetloader = []
        for x in range(num_rounds):
            start_idx = (x * subset_samples) % len(subset)
            end_idx = (start_idx + subset_samples) % len(subset)

            if start_idx < end_idx:
                selected_indices = range(start_idx, end_idx)
            else:
                selected_indices = list(range(start_idx, len(subset))) + list(
                    range(0, end_idx)
                )

            subsetloader.append(
                DataLoader(
                    subset.select(selected_indices),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )
    else:
        subsetloader = DataLoader(
            subset.select(range(subset_samples)), batch_size=batch_size, shuffle=True
        )

    if cfg.noise:
        if "fmnist" in cfg.dataset:
            noise_dataset = RandomNoiseDataset(subset_samples, (1, 28, 28), cfg)
        elif "cifar" in cfg.dataset:
            noise_dataset = RandomNoiseDataset(subset_samples, (3, 32, 32), cfg)
        subsetloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    return trainloaders, valloaders, subsetloader


def load_attack_sets_from_noise(
    input_shape, num_labels, sample_per_label, batch_size, cfg
):
    attack_loaders = []
    for i in range(num_labels):
        if cfg.dataset == "shakespeare":
            dataset = RandomTextDataset(
                sample_per_label, cfg.seq_length, cfg.vocab_size, cfg, label=i
            )
        else:
            dataset = RandomNoiseDataset(sample_per_label, input_shape, cfg, label=i)
        attack_loaders.append(DataLoader(dataset, batch_size=batch_size))
    return attack_loaders


class RandomNoiseDataset(Dataset):
    def __init__(self, size, image_shape, cfg, label=None):
        self.size = size
        self.image_shape = image_shape
        self.label = label
        self.cfg = cfg

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(self.image_shape) * 0.5 + 0.5
        if self.label is None:
            label = torch.randint(0, 10, (1,)).item()  # Random label between 0 and 9
        else:
            label = self.label
        return {"img": image, self.cfg.image_label: label}


class RandomTextDataset(Dataset):
    """Random noise dataset for text (for canary attacks)"""

    def __init__(self, size, seq_length, vocab_size, cfg, label=None):
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.cfg = cfg
        self.label = label

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sequence = torch.randint(0, self.vocab_size, (self.seq_length,))
        if self.label is None:
            target = torch.randint(0, self.vocab_size, (1,)).item()
        else:
            target = self.label
        return {"input": sequence, self.cfg.text_label: target}


def sequences_to_dataset_dict(sequences, targets, cfg):
    return {
        cfg.text_input: sequences,  # List of sequences
        cfg.text_label: targets,  # List of target indices
    }


def create_shakespeare_transform(cfg):
    """
    Creates a transform function to convert Shakespeare data to tensors.
    """

    def transform_fn(batch):
        batch[cfg.text_input] = torch.tensor(batch[cfg.text_input])

        batch[cfg.text_label] = torch.tensor(batch[cfg.text_label])

        return batch

    return transform_fn


def load_shakespeare_subsets_offline(cfg):
    num_clients = cfg.num_clients
    num_rounds = cfg.num_rounds
    batch_size = cfg.batch_size

    if cfg.iid:
        partitioner = IidPartitioner(num_clients)
    else:
        partitioner = DirichletPartitioner(
            num_clients,
            partition_by=cfg.text_label,
            alpha=cfg.dirichlet_alpha,
            min_partition_size=0,
        )

    train_x, train_y = load_data("{}train_data.pkl".format(cfg.path_to_local_dataset))
    test_x, test_y = load_data("{}test_data.pkl".format(cfg.path_to_local_dataset))

    try:
        vocab_path = "{}vocab.pkl".format(cfg.path_to_local_dataset)
        with open(vocab_path, "rb") as f:
            vocab_info = pickle.load(f)
        cfg.vocab_size = vocab_info["vocab_size"]
        print(f"  Vocabulary size: {cfg.vocab_size}")
    except FileNotFoundError:
        print("  Warning: vocab.pkl not found, using vocab_size from config")

    train_dict = sequences_to_dataset_dict(train_x, train_y, cfg)
    partitioner.dataset = datasets.Dataset.from_dict(train_dict)

    apply_transforms = create_shakespeare_transform(cfg)

    trainloaders = []
    valloaders = []

    for partition_id in range(num_clients):
        partition_client = partitioner.load_partition(partition_id)
        partition_client = partition_client.with_transform(apply_transforms)

        partition_client = partition_client.train_test_split(train_size=0.8)

        train_samples = min(cfg.train_samples, len(partition_client["train"]))
        partition_train = partition_client["train"].select(range(train_samples))
        test_samples = min(cfg.test_samples, len(partition_client["test"]))
        partition_val = partition_client["test"].select(range(test_samples))

        print(
            "Client {}: {} train samples, {} test samples".format(
                partition_id, len(partition_train), len(partition_val)
            )
        )

        trainloaders.append(
            DataLoader(partition_train, batch_size=batch_size, shuffle=True)
        )
        valloaders.append(
            DataLoader(partition_val, batch_size=batch_size, shuffle=True)
        )

    test_dict = sequences_to_dataset_dict(test_x, test_y, cfg)
    subset = datasets.Dataset.from_dict(test_dict)
    subset = subset.with_transform(apply_transforms)

    subset_samples = min(cfg.subset_samples, len(subset))

    if cfg.dynamic_canary:
        subsetloader = []
        for x in range(num_rounds):
            start_idx = (x * subset_samples) % len(subset)
            end_idx = (start_idx + subset_samples) % len(subset)

            if start_idx < end_idx:
                selected_indices = range(start_idx, end_idx)
            else:
                selected_indices = list(range(start_idx, len(subset))) + list(
                    range(0, end_idx)
                )

            subsetloader.append(
                DataLoader(
                    subset.select(selected_indices),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )
        print(f"Created {len(subsetloader)} dynamic canary loaders")
    else:
        subsetloader = DataLoader(
            subset.select(range(subset_samples)), batch_size=batch_size, shuffle=True
        )
        print("Created static canary loader")

    if cfg.noise:
        print(
            f"Replacing canary with random noise (vocab_size={cfg.vocab_size}, seq_length={cfg.seq_length})"
        )
        if cfg.dynamic_canary:
            subsetloader = []
            for x in range(num_rounds):
                noise_dataset = RandomTextDataset(
                    subset_samples, cfg.seq_length, cfg.vocab_size, cfg
                )
                subsetloader.append(
                    DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)
                )
        else:
            noise_dataset = RandomTextDataset(
                subset_samples, cfg.seq_length, cfg.vocab_size, cfg
            )
            subsetloader = DataLoader(
                noise_dataset, batch_size=batch_size, shuffle=False
            )

    return trainloaders, valloaders, subsetloader


def load_data(filename):
    """Load pickle file - works for both image and text data"""
    with open(filename, "rb") as f:
        return pickle.load(f)


_SUBSET_CACHE = None


def load_partition_offline(cfg, partition_id: int):
    """Load only ONE client's train/val loaders (image datasets, offline PKL)."""
    num_clients = int(cfg.num_clients)
    partition_id = int(partition_id)
    batch_size = int(cfg.batch_size)

    if cfg.iid:
        partitioner = IidPartitioner(num_clients)
    else:
        partitioner = DirichletPartitioner(
            num_clients,
            partition_by=cfg.image_label,
            alpha=float(cfg.dirichlet_alpha),
            min_partition_size=0,
        )

    train_x, train_y = load_data(f"{cfg.path_to_local_dataset}train_data.pkl")
    test_x, test_y = load_data(f"{cfg.path_to_local_dataset}test_data.pkl")

    train_pil_images = images_to_pil(train_x, cfg.dataset)

    train_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in train_y]
    test_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in test_y]

    partitioner.dataset = datasets.Dataset.from_dict(
        {cfg.image_name: train_pil_images, cfg.image_label: train_y}
    )

    apply_transforms = create_apply_transforms(cfg)

    part = partitioner.load_partition(partition_id)
    part = part.with_transform(apply_transforms)
    part = part.train_test_split(train_size=0.8)

    train_samples = min(int(cfg.train_samples), len(part["train"]))
    test_samples = min(int(cfg.test_samples), len(part["test"]))

    trainset = part["train"].select(range(train_samples))
    valset = part["test"].select(range(test_samples))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    return trainloader, valloader


def load_local_data_offline(cfg, data_path: str):
    """Deployment mode: node already has its own local dataset on disk."""
    if not data_path.endswith("/"):
        data_path += "/"

    batch_size = int(cfg.batch_size)

    train_x, train_y = load_data(f"{data_path}train_data.pkl")
    test_x, test_y = load_data(f"{data_path}test_data.pkl")

    train_pil = images_to_pil(train_x, cfg.dataset)

    train_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in train_y]
    test_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in test_y]

    apply_transforms = create_apply_transforms(cfg)

    ds_train = datasets.Dataset.from_dict(
        {cfg.image_name: train_pil, cfg.image_label: train_y}
    ).with_transform(apply_transforms)
    split = ds_train.train_test_split(train_size=0.8)
    train_samples = min(int(cfg.train_samples), len(split["train"]))
    val_samples = min(int(cfg.test_samples), len(split["test"]))
    trainset = split["train"].select(range(train_samples))
    valset = split["test"].select(range(val_samples))

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)
    return trainloader, valloader


def get_subsetloaders_offline(cfg):
    global _SUBSET_CACHE
    if _SUBSET_CACHE is not None:
        return _SUBSET_CACHE

    num_rounds = cfg.num_rounds
    batch_size = cfg.batch_size

    test_x, test_y = load_data(f"{cfg.path_to_local_dataset}test_data.pkl")
    test_pil_images = images_to_pil(test_x, cfg.dataset)
    test_y = [y[0] if isinstance(y, (list, tuple)) else int(y) for y in test_y]

    apply_transforms = create_apply_transforms(cfg)
    subset = datasets.Dataset.from_dict(
        {cfg.image_name: test_pil_images, cfg.image_label: test_y}
    ).with_transform(apply_transforms)

    subset_samples = min(cfg.subset_samples, len(subset))

    if cfg.dynamic_canary:
        subsetloaders = []
        for r in range(num_rounds):
            start_idx = (r * subset_samples) % len(subset)
            end_idx = (start_idx + subset_samples) % len(subset)
            if start_idx < end_idx:
                idxs = range(start_idx, end_idx)
            else:
                idxs = list(range(start_idx, len(subset))) + list(range(0, end_idx))
            subsetloaders.append(
                DataLoader(subset.select(idxs), batch_size=batch_size, shuffle=True)
            )
    else:
        subsetloaders = DataLoader(
            subset.select(range(subset_samples)), batch_size=batch_size, shuffle=True
        )

    if cfg.noise:
        if "fmnist" in cfg.dataset:
            noise_dataset = RandomNoiseDataset(subset_samples, (1, 28, 28), cfg)
        else:
            noise_dataset = RandomNoiseDataset(subset_samples, (3, 32, 32), cfg)
        subsetloaders = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)

    _SUBSET_CACHE = subsetloaders
    return subsetloaders
