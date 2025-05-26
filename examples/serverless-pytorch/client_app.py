"""Client app for serverless federated learning with PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import OrderedDict

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from flwr.serverless import AsyncFederatedNode, SyncFederatedNode, InMemoryFolder

from net import ResNet18


def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def load_data(batch_size=32):
    """Load CIFAR-10 data."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load dataset
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size)
    
    return trainloader, valloader


def train(model, trainloader, valloader, node, epochs=5):
    """Train the model using federated learning."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 100 == 99:
                print(f'Node {node.node_id} - Epoch {epoch + 1}, '
                      f'Step [{batch_idx + 1}/{len(trainloader)}], '
                      f'Loss: {running_loss / 100:.3f}, '
                      f'Acc: {100. * correct / total:.3f}%')
                running_loss = 0.0
        
        # Federate the model
        weights = [p.detach().cpu().numpy() for p in model.parameters()]
        flwr_parameters = ndarrays_to_parameters(weights)
        
        # Update parameters through federation
        updated_parameters, metrics = node.update_parameters(
            local_parameters=flwr_parameters,
            num_examples=total,
            metrics={'loss': running_loss / len(trainloader), 'accuracy': correct / total},
            epoch=epoch
        )
        
        # Update model with federated weights
        if updated_parameters:
            new_weights = parameters_to_ndarrays(updated_parameters)
            for param, new_weight in zip(model.parameters(), new_weights):
                param.data = torch.tensor(new_weight).to(device)
        
        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in valloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(target.view_as(pred)).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(valloader)
        val_accuracy = val_correct / val_total
        
        print(f'Node {node.node_id} - Epoch {epoch + 1} - '
              f'Val Loss: {val_loss:.3f}, Val Acc: {100. * val_accuracy:.3f}%')


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
        self.storage_backend = InMemoryFolder()
        self.strategy = FedAvg()
        self.node = AsyncFederatedNode(
            shared_folder=self.storage_backend,
            strategy=self.strategy,
            node_id=self.node_id
        )

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.node,
            self.local_epochs
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
    trainloader, valloader = load_data(batch_size)
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

def main():
    # Create model
    model = ResNet18(small_resolution=True)
    
    # Load data
    trainloader, valloader = load_data()
    
    # Create shared folder for model storage
    shared_folder = InMemoryFolder()
    
    # Create federated node (async or sync)
    node_id = "client_1"  # In a real scenario, this would be unique per client
    strategy = FedAvg()
    node = AsyncFederatedNode(
        shared_folder=shared_folder,
        strategy=strategy,
        node_id=node_id
    )
    
    # Train the model
    train(model, trainloader, valloader, node)

if __name__ == "__main__":
    main() 