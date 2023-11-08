import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset, ConcatDataset
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter

# Define the root directory where your dataset is located
dataset_root = "data/pacs_data/"


# List the subdirectories (folders) in the parent folder
data_kinds = [
    os.path.join(dataset_root, d)
    for d in os.listdir(dataset_root)
    if os.path.isdir(os.path.join(dataset_root, d))
]


def make_dataloaders(dataset_kinds=data_kinds, k=2, batch_size=32, verbose=False):
    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    trainloaders = []
    testloaders = []
    val_subsets = []
    for idx, data_kind in enumerate(dataset_kinds):
        dataset = datasets.ImageFolder(root=data_kind, transform=transform)
        for fold, (train_indices, test_indices) in enumerate(
            skf.split(range(len(dataset)), dataset.targets), start=1
        ):
            # Split the dataset into train and test subsets
            train_dataset = Subset(dataset, train_indices)
            test_dataset = Subset(dataset, test_indices)
            if fold == 1:
                # Create DataLoaders for train and test sets
                train_loader_tmp = DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=True
                )
                test_loader_tmp = DataLoader(test_dataset, batch_size=batch_size)
                if verbose:
                    print(
                        f"Fold {fold}: Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}"
                    )
                    test_labels = []

                    # Iterate through the test DataLoader to collect labels
                    for images, labels in test_loader_tmp:
                        test_labels.extend(labels.tolist())

                    # Calculate and print the class label frequency
                    label_counts = Counter(test_labels)

                    for label, count in label_counts.items():
                        print(f"Class {label}: {count} samples")

                trainloaders.append(train_loader_tmp)
                testloaders.append(test_loader_tmp)
            else:
                val_subsets.append(test_dataset)
    combined_val_dataset = ConcatDataset(val_subsets)
    valloader = DataLoader(combined_val_dataset, batch_size=batch_size, shuffle=True)

    return (trainloaders, testloaders, valloader)


if __name__ == "__main__":
    aq, bq, cq = make_dataloaders(verbose=True)

    cc = 0
    for feat, label in cq:
        print(len(label))
        cc += len(label)
