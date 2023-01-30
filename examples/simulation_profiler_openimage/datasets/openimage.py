from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
import pandas as pd
import numpy as np
import torch

# Adapted from: https://github.com/SymbioticLab/FedScale/blob/a6ce9f1c15287b8a9704ef6ce2bf66508bcd3340/fedscale/dataloaders/utils_data.py#L107
train_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


class OpenImage(Dataset):
    def __init__(self, cid_csv_root: Path, img_root: Path):
        self.cid_csv_root = cid_csv_root
        self.img_root = img_root
        self.dataset_type = "train"

        self.transforms = {}
        self.transforms["train"] = train_transform
        self.transforms["test"] = test_transform

        self.data: pd.DataFrame

    def load_client(self, *, cid: str, dataset_type: str = "train"):
        filename = self.cid_csv_root / dataset_type / f"{cid}.csv"
        self.data = pd.read_csv(
            filename,
            # engine="pyarrow",
            dtype={"images": "string", "labels": np.int64},
            names=["images", "labels"],
        )

    def __getitem__(self, index: int):
        img_name, target = self.data.iloc[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(self.img_root / self.dataset_type / img_name)

        # avoid channel error
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transforms[self.dataset_type] is not None:
            img = self.transforms[self.dataset_type](img)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    img_root = Path("/datasets/FedScale/openImg/")
    cid_csv_root = Path("/datasets/FedScale/openImg/client_data_mapping/clean_ids")
    train_dataset = OpenImage(cid_csv_root, img_root)

    train_dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )

    for img, lbl in train_dataloader:
        print(img.shape)
        print(lbl)
