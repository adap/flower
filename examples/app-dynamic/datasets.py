from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import datasets, transforms
from torchvision.transforms import Compose, Normalize, ToTensor


def load_cifar_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def load_mnist_data():
    """Load MNIST (training and test set)."""
    # MNIST images are grayscale; normalize with mean and std of MNIST
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load training and test sets
    trainset = datasets.MNIST("./data", train=True, download=True, transform=trf)
    testset = datasets.MNIST("./data", train=False, download=True, transform=trf)

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = DataLoader(testset, batch_size=32, shuffle=False)

    return train_loader, test_loader
