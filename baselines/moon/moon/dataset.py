"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

# https://github.com/QinbinLi/MOON/blob/main/datasets.py

import logging

import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable


normalize_c10 = transforms.Normalize(
    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
)

normalize_c100 = transforms.Normalize(
    mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
    std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404],
)

def get_train_transforms(dataset_name: str):
    if dataset_name == "uoft-cs/cifar10":
        return transforms.Compose(
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
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_c10,
            ]
        )
    elif dataset_name == "uoft-cs/cifar100":

        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize_c100,
            ]
        )

    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` not recognized.")
    

def get_eval_transforms(dataset_name: str):

    if dataset_name == "uoft-cs/cifar10":
        return transforms.Compose([transforms.ToTensor(), normalize_c10])
    elif dataset_name == "uoft-cs/cifar100":
        return transforms.Compose([transforms.ToTensor(), normalize_c100])

    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` not recognized.")


def get_apply_transforms_fn(transforms, dataset_name):

    if dataset_name == "uoft-cs/cifar10":
        label = "label"
    elif dataset_name == "uoft-cs/cifar100":
        label = "fine_label"
    else:
        raise NotImplementedError(f"Dataset `{dataset_name}` not recognized.")
     
    def apply_transforms(batch):
        # For CIFAR-10 the "img" column contains the images we want to
        # apply the transforms to
        batch["img"] = [transforms(img) for img in batch["img"]]
        # map to a common column just to implify training loop
        # Note "label" doesn't exist in CIFAR-100
        batch["label"] = batch[label]
        return batch

    return apply_transforms