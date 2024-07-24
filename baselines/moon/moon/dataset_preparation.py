"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""

import torchvision.transforms as transforms

import torch.nn.functional as F
from torch.autograd import Variable

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner


fds = None

def get_dataset(dataset_name: str, dirichlet_alpha: float, num_partitions: int, partition_by:str) -> FederatedDataset:

    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(num_partitions=num_partitions, alpha=dirichlet_alpha, partition_by=partition_by)
        fds = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    
    return fds


def get_data_transforms(dataset_name):
    """Get dataset transforms"""
    if dataset_name == "uoft-cs/cifar10":
        normalize = transforms.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(
                    lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4),
                        mode="reflect",
                    ).data.squeeze()
                ),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        # data prep for test set
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    elif dataset_name == "uoft-cs/cifar100":

        normalize = transforms.Normalize(
            mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
        )

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
            ]
        )
        # data prep for test set
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        raise ValueError("Only datasets `uoft-cs/cifar10` and `uoft-cs/cifar100` are supported")
    
    return transform_train, transform_test


def get_transforms_apply_fn(transforms):

    def apply_transforms(batch):
        # For CIFAR-10 the "img" column contains the images we want to apply the transforms to
        batch["img"] = [transforms(img) for img in batch["img"]]
        return batch

    return apply_transforms
