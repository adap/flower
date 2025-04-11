"""Task module for serverless federated learning with PyTorch."""

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms



def load_data(partition_id, num_partitions, batch_size):
    """Load partition of CIFAR10 data."""
    # Define data transforms
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    
    # Load the full dataset
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Calculate partition size
    partition_size = len(dataset) // num_partitions
    
    # Create partitions
    partitions = random_split(dataset, [partition_size] * num_partitions)
    
    # Get the partition for this client
    partition = partitions[partition_id]
    
    # Split into train and validation
    train_size = int(0.8 * len(partition))
    val_size = len(partition) - train_size
    train_dataset, val_dataset = random_split(partition, [train_size, val_size])
    
    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return trainloader, valloader

