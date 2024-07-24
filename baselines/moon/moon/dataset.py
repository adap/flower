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

# TODO: Delete this file

import logging
import os

import numpy as np
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from torchvision.datasets import CIFAR10, CIFAR100

# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

IMG_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
)


class CIFAR10Sub(data.Dataset):
    """CIFAR-10 dataset with idxs."""

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_sub_dataset__()

    def __build_sub_dataset__(self):
        """Build sub dataset given idxs."""
        cifar_dataobj = CIFAR10(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        if torchvision.__version__ == "0.2.1":
            if self.train:
                # pylint: disable=redefined-outer-name
                data, target = cifar_dataobj.train_data, np.array(
                    cifar_dataobj.train_labels
                )
            else:
                # pylint: disable=redefined-outer-name
                data, target = cifar_dataobj.test_data, np.array(
                    cifar_dataobj.test_labels
                )
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """Get item by index.

        Args:
            index (int): Index.

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Length.

        Returns
        -------
            int: length of data
        """
        return len(self.data)


class CIFAR100Sub(data.Dataset):
    """CIFAR-100 dataset with idxs."""

    def __init__(
        self,
        root,
        dataidxs=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_sub_dataset__()

    def __build_sub_dataset__(self):
        """Build sub dataset given idxs."""
        cifar_dataobj = CIFAR100(
            self.root, self.train, self.transform, self.target_transform, self.download
        )

        if torchvision.__version__ == "0.2.1":
            if self.train:
                # pylint: disable=redefined-outer-name
                data, target = cifar_dataobj.train_data, np.array(
                    cifar_dataobj.train_labels
                )
            else:
                data, target = cifar_dataobj.test_data, np.array(
                    cifar_dataobj.test_labels
                )  # pylint: disable=redefined-outer-name
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """Get item by index.

        Args:
            index (int): Index.

        Returns
        -------
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Length.

        Returns
        -------
            int: length of data
        """
        return len(self.data)


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0):
    """Get dataloader for a given dataset."""
    if dataset == "cifar10":
        dl_obj = CIFAR10Sub
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
                transforms.ColorJitter(brightness=noise_level),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
        # data prep for test set
        transform_test = transforms.Compose([transforms.ToTensor(), normalize])

    elif dataset == "cifar100":
        dl_obj = CIFAR100Sub

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
    if dataset == "cifar10" and os.path.isdir(
        os.path.join(datadir, "cifar-10-batches-py")
    ):
        download = False
    elif dataset == "cifar100" and os.path.isdir(
        os.path.join(datadir, "cifar-100-python")
    ):
        download = False
    else:
        download = True
    train_ds = dl_obj(
        datadir,
        dataidxs=dataidxs,
        train=True,
        transform=transform_train,
        download=download,
    )
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=download)

    train_dl = data.DataLoader(
        dataset=train_ds, batch_size=train_bs, drop_last=True, shuffle=True
    )
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False)

    return train_dl, test_dl, train_ds, test_ds
