from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomResizedCrop,
    Resize,
    CenterCrop,
)

from flwr_datasets import FederatedDataset


def get_dataset_with_partitions(num_partitions: int):
    """Get CIFAR-100 datasets and partition it.

    Return partitioned dataset as well as the whole test set.
    """

    # Get CIFAR-100 and divide it into 20 IID partitions
    c100_fds = FederatedDataset(
        dataset="cifar100", partitioners={"train": num_partitions}
    )

    centralized_testset = c100_fds.load_full("test")
    return c100_fds, centralized_testset


def apply_eval_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch


def apply_transforms(batch):
    """Apply a very standard set of image transforms."""
    transforms = Compose(
        [
            RandomResizedCrop((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch["img"] = [transforms(img) for img in batch["img"]]
    return batch
