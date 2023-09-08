from torchvision.datasets import CIFAR10

from flwr_datasets import FederatedDataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

if __name__ == "__main__":

    fds = FederatedDataset(dataset="cifar", partitioners={"train": 100})
    partition = fds.load_partition(10, "train")

    # Specify Transforms
    transforms = ToTensor()
    partition_torch = partition.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")
    dataloader = DataLoader(partition_torch, batch_size=16)

    # You can divide the dataset
    partition_train_test = partition.train_test_split(test_size=0.2)
    partition_train = partition_train_test["train"]
    partition_test = partition_train_test["test"]

    # Alternatively
    partition_len = len(partition)
    partition_train = partition[:int(0.8 * partition_len)]
    partition_test = partition[int(0.8 * partition_len):]

    # Each of the train and test partitions needs to be handled separately now
    partition_train_torch = partition_train.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")
    partition_test_torch = partition_test.map(
        lambda img: {"img": transforms(img)}, input_columns="img"
    ).with_format("torch")

    train_dataloader = DataLoader(partition_train_torch, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(partition_test_torch, batch_size=16)

    # This is how it's done using PyTorch only
    train_dataset_from_pytorch = CIFAR10("./data", train=True, download=True,
                                         transform=transforms)
    all_from_pytorch_dataloader = DataLoader(train_dataset_from_pytorch,
                                             batch_size=16,
                                             shuffle=True)

    # How do you iterate through the data in when using only PyTorch
    for batch in all_from_pytorch_dataloader:
        images, labels = batch
        # Equivalently
        images, labels = batch[0], batch[1]

        # Let's inspect images
        print("Shape of images:")
        print(images.shape)
        print("Data Type of images:")
        print(type(images))
        print("First image from the batch - values:")
        print(images[0])

        # Let's inspect labels
        print("Shape of labels:")
        print(labels.shape)
        print("Data Type of labels:")
        print(type(labels))
        print("First label from the batch - values:")
        print(labels[0])
        break

    # How do you iterate through the data in when using Flower Datasets
    for batch in dataloader:
        images, labels = batch["img"], batch["label"]

        # Let's inspect images
        print("Shape of images:")
        print(images.shape)
        print("Data Type of images:")
        print(type(images))
        print("First image from the batch - values:")
        print(images[0])

        # Let's inspect labels
        print("Shape of labels:")
        print(labels.shape)
        print("Data Type of labels:")
        print(type(labels))
        print("First label from the batch - values:")
        print(labels[0])
        break

    # Note that the value are not the same yet the dimensions and data types match
    # Don't worry about the values, it's just the order of the samples
