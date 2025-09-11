"""Quantum Federated Learning Client with PennyLane and Flower."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from pennylane_example.task import QuantumNet, get_weights, set_weights, load_data, train, test


class QuantumFlowerClient(NumPyClient):
    """Flower client implementing quantum neural network training."""
    
    def __init__(self, trainloader, valloader, local_epochs: int, learning_rate: float):
        self.net = QuantumNet(num_classes=10)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Client initialized with device: {self.device}")
        print(f"Training data size: {len(trainloader.dataset)}")
        print(f"Validation data size: {len(valloader.dataset)}")

    def fit(self, parameters, config):
        """Train the quantum neural network with local data."""
        print(f"Starting local training for {self.local_epochs} epochs...")
        
        # Set model parameters received from server
        set_weights(self.net, parameters)
        
        # Train the model
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.learning_rate,
            self.device
        )
        
        print(f"Training completed. Train loss: {results['train_loss']:.4f}, "
              f"Val accuracy: {results['val_accuracy']:.2f}%")
        
        # Return updated weights and metrics
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the quantum neural network on local validation data."""
        print("Starting local evaluation...")
        
        # Set model parameters received from server
        set_weights(self.net, parameters)
        
        # Evaluate the model
        loss, accuracy = test(self.net, self.valloader, self.device)
        
        print(f"Evaluation completed. Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Flower client for quantum federated learning."""
    
    # Read configuration from context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    
    # Read hyperparameters from run config
    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"] 
    learning_rate = context.run_config["learning-rate"]
    
    print(f"Client {partition_id}/{num_partitions} starting...")
    print(f"Hyperparameters - Batch size: {batch_size}, "
          f"Local epochs: {local_epochs}, Learning rate: {learning_rate}")
    
    # Load and partition data
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    
    # Create and return client instance
    return QuantumFlowerClient(
        trainloader, valloader, local_epochs, learning_rate
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
