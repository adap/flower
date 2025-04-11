"""Client app for serverless federated learning with PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from collections import OrderedDict

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from flwr.serverless.federated_node.async_federated_node import AsyncFederatedNode
from flwr.serverless.shared_folder.local_folder import LocalFolder

from net import ResNet18


def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


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


def train(net, trainloader, valloader, epochs, learning_rate, device, node=None, node_id=None):
    """Train the model on the training set."""
    print(f"Training on node {node_id}. Device: {device}")
    net.to(device)  # move model to GPU if available
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    
    net.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 99:
                accuracy = correct / total
                print(f'[Node {node_id}] Epoch [{epoch + 1}/{epochs}], '
                      f'Step [{batch_idx + 1}/{len(trainloader)}], '
                      f'Loss: {running_loss / 100:.3f}, '
                      f'Acc: {100. * accuracy:.3f}%')
                running_loss = 0.0
        
        print(f"Finished local training for epoch {epoch + 1} of {epochs} on node {node_id}")
        
        # If we have a node, perform federation after each epoch
        if node is not None:
            print(f"Federating on node {node_id}")
            # Convert torch model weights to flwr parameters
            flwr_parameters = ndarrays_to_parameters(get_weights(net))
            
            # Perform federation using the node
            accuracy = correct / total
            updated_parameters, metrics = node.update_parameters(
                local_parameters=flwr_parameters,
                num_examples=len(trainloader.dataset),
                metrics={
                    "loss": loss.item(),
                    "accuracy": accuracy,
                },
                epoch=epoch,
            )
            
            # Convert flwr parameters back to torch model weights
            np_parameters = parameters_to_ndarrays(updated_parameters)
            set_weights(net, np_parameters)
            print(f"Updated model weights on node {node_id}")
    
    # Evaluate on validation set
    print(f"Evaluating on node {node_id}")
    val_loss, val_acc = test(net, valloader, device)
    print(f"Finished evaluation on node {node_id}")
    
    results = {
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    print(f"Results on node {node_id}: {results}")
    return results


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)  # move model to GPU if available
    criterion = nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate, node_id):
        self.net = ResNet18(small_resolution=True)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Client {node_id} initialized on device: {self.device}")
        self.node_id = node_id
        
        # Create AsyncFederatedNode for this client
        self.storage_backend = LocalFolder("./shared_folder")
        self.strategy = FedAvg()
        self.node = AsyncFederatedNode(
            shared_folder=self.storage_backend,
            strategy=self.strategy
        )

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
            self.node,
            self.node_id
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(
        trainloader, 
        valloader, 
        local_epochs, 
        learning_rate,
        partition_id
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn) 