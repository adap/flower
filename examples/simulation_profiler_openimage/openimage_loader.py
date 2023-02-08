from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# borrowed from: https://github.com/SymbioticLab/FedScale/blob/a6ce9f1c15287b8a9704ef6ce2bf66508bcd3340/fedscale/dataloaders/utils_data.py#L107
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
    def __init__(self, root: Path, cid: str, dataset: str = "train", transform=None):

        self.root = root
        self.transform = transform
        self.data_type = dataset  # 'train', 'test', 'val'
        self.path = root / dataset
        self.data = []
        self.targets = []
        with open(
            root / "client_data_mapping" / "clean_ids" / dataset / f"{cid}.csv", "r"
        ) as f:
            for line in f:
                img_name, target_idx = line.strip("\n").split(",")
                self.data.append(img_name)
                self.targets.append(int(target_idx))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_name, target = self.data[index], torch.tensor(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(self.path / img_name)

        # avoid channel error
        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    path = Path("/datasets/FedScale/openImg/")
    train_data = OpenImage(
        root=path, cid="99", dataset="train", transform=train_transform
    )
    print(train_data)

    train_dataloader = DataLoader(
        train_data, batch_size=128, shuffle=True, pin_memory=True, num_workers=4
    )

    for img, lbl in train_data:
        print(img.shape)
        print(lbl)
