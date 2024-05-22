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
    """Get Oxford Flowers datasets and partition it.

    Return partitioned dataset as well as the whole test set.
    """

    # Get Oxford Flowers-102 and divide it into 20 IID partitions
    ox_flowers_fds = FederatedDataset(
        dataset="nelorth/oxford-flowers", partitioners={"train": num_partitions}
    )

    centralized_testset = ox_flowers_fds.load_split("test")
    return ox_flowers_fds, centralized_testset


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
    batch["image"] = [transforms(img) for img in batch["image"]]
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
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch
