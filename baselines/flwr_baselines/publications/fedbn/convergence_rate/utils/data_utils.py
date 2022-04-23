"""This code creates 10 different partitions of each datasets."""
import os
import sys

import numpy as np
from PIL import Image  # type: ignore
from torch.utils.data import Dataset

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)


class DigitsDataset(Dataset):
    """Split datasets."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        data_path,
        channels,
        percent=0.1,
        filename=None,
        train=True,
        transform=None,
    ):
        if filename is None:
            if train:
                if percent >= 0.1:
                    for part in range(int(percent * 10)):
                        if part == 0:
                            self.images, self.labels = np.load(
                                os.path.join(
                                    data_path,
                                    f"partitions/train_part{part}.pkl",
                                ),
                                allow_pickle=True,
                            )
                        else:
                            images, labels = np.load(
                                os.path.join(
                                    data_path,
                                    f"partitions/train_part{part}.pkl",
                                ),
                                allow_pickle=True,
                            )
                            self.images = np.concatenate([self.images, images], axis=0)
                            self.labels = np.concatenate([self.labels, labels], axis=0)
                else:
                    self.images, self.labels = np.load(
                        os.path.join(data_path, "partitions/train_part0.pkl"),
                        allow_pickle=True,
                    )
                    data_len = int(self.images.shape[0] * percent * 10)
                    self.images = self.images[:data_len]
                    self.labels = self.labels[:data_len]
            else:
                self.images, self.labels = np.load(
                    os.path.join(data_path, "test.pkl"), allow_pickle=True
                )
        else:
            self.images, self.labels = np.load(
                os.path.join(data_path, filename), allow_pickle=True
            )

        self.transform = transform
        self.channels = channels
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.channels == 1:
            image = Image.fromarray(image, mode="L")
        elif self.channels == 3:
            image = Image.fromarray(image, mode="RGB")
        else:
            raise ValueError(f"{self.channels} channel is not allowed.")

        if self.transform is not None:
            image = self.transform(image)

        return image, label
