"""Serverless PyTorch Federated Learning Simulation.

This example demonstrates proper federated learning practices:
1. Each node calls update_parameters to produce federated model
2. Only node 0 performs centralized evaluation on federated model
3. Skewed data distribution across nodes
4. Federated aggregation using Flower's strategy (FedAvg)
5. Comprehensive experiment tracking following FL best practices

Note: In proper FL, each node produces a federated model through update_parameters,
but only one node (node 0) performs centralized evaluation to avoid redundancy.
"""
from dataclasses import dataclass
from typing import Dict, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from flwr.serverless import FederatedExperimentRunner
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from net import ResNet18


@dataclass
class Metrics:
    """Metrics for training, validation, or testing phase."""
    loss: float
    accuracy: float
    samples: int
    interrupted: bool = False
    phase: str = "train"  # "train", "val", or "test"


def train_node(model, data_loaders, node, config, test_loader=None):
    """Train a single node for multiple epochs with federated aggregation."""
    trainloader, valloader = data_loaders
    epochs = config.get('epochs', 5)
    
    # Update steps_per_epoch in config
    config['steps_per_epoch'] = len(trainloader)
    
    # Setup training components
    model, optimizer, criterion, scheduler, device = setup_training_components(model, config)
    
    print(f"Training node {node.node_id} for {epochs} epochs "
          f"(max_lr={config.get('learning_rate', 0.01)}, momentum={config.get('momentum', 0.9)})")
    if test_loader is not None:
        print(f"  Node {node.node_id} will perform centralized evaluation")
    
    epoch_metrics = []
    
    for epoch in range(epochs):
        # Check for interruption before each epoch
        if check_interruption(config):
            print(f"  Node {node.node_id} - Training interrupted at epoch {epoch + 1}")
            break
        
        # Training phase
        train_metrics = train_single_epoch(
            model, trainloader, optimizer, criterion, scheduler, device, node.node_id, epoch, config
        )
        if train_metrics.interrupted:
            break
        
        # Validation phase
        val_metrics = validate_single_epoch(
            model, valloader, criterion, device
        )
        
        # Get current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Print epoch status
        print_epoch_status(node.node_id, epoch, epochs, train_metrics, val_metrics, current_lr)
        
        #########################################################
        # Federated aggregation: the core logic.
        #########################################################
        parameters = get_model_parameters_as_flower_params(model)
        federated_params, federated_metrics = node.update_parameters(
            parameters, 
            num_examples=train_metrics.samples,
            metrics={
                'train_loss': train_metrics.loss,
                'train_accuracy': train_metrics.accuracy,
                'train_samples': train_metrics.samples,
            },
            epoch=epoch
        )
        
        # Update model with federated parameters
        update_model_with_federated_params(model, federated_params)

        ## Metrics
        # Centralized evaluation (only for node 0)
        test_metrics = None
        if test_loader is not None:
            test_metrics = perform_centralized_evaluation(
                model, test_loader, criterion, device, node.node_id
            )
        
        # Store epoch metrics
        epoch_metric = create_epoch_metrics(
            epoch, train_metrics, val_metrics, federated_metrics, 
            test_metrics, current_lr
        )
        epoch_metrics.append(epoch_metric)
        
        # Final interruption check after epoch
        if check_interruption(config):
            print(f"  Node {node.node_id} - Training interrupted after epoch {epoch + 1}")
            break
    
    # Training completion status
    if check_interruption(config):
        print(f"Node {node.node_id} training interrupted (completed {len(epoch_metrics)}/{epochs} epochs)")
    else:
        print(f"Node {node.node_id} completed training and federated aggregation")
    
    return model


def create_epoch_metrics(epoch: int, train_metrics: Metrics, val_metrics: Metrics, 
                        federated_metrics: Dict, test_metrics: Optional[Metrics] = None, 
                        learning_rate: Optional[float] = None) -> Dict:
    """Create a dictionary of metrics for the current epoch."""
    metrics = {
        'epoch': epoch + 1,
        'train_loss': train_metrics.loss,
        'train_accuracy': train_metrics.accuracy,
        'train_samples': train_metrics.samples,
        'val_loss': val_metrics.loss,
        'val_accuracy': val_metrics.accuracy,
        'val_samples': val_metrics.samples,
        'federated_metrics': federated_metrics,
        'learning_rate': learning_rate
    }
    
    if test_metrics:
        metrics.update({
            'test_loss': test_metrics.loss,
            'test_accuracy': test_metrics.accuracy,
            'test_samples': test_metrics.samples
        })
    
    return metrics

def print_epoch_status(node_id: str, epoch: int, epochs: int, 
                      train_metrics: Metrics, val_metrics: Metrics, 
                      learning_rate: float):
    """Print the status of the current epoch."""
    print(f"  Node {node_id} - Epoch {epoch + 1}/{epochs}: "
          f"Train Loss: {train_metrics.loss:.4f}, Train Acc: {100*train_metrics.accuracy:.2f}%, "
          f"Val Loss: {val_metrics.loss:.4f}, Val Acc: {100*val_metrics.accuracy:.2f}%, "
          f"LR: {learning_rate:.6f}")


def setup_training_components(model, config):
    """Set up optimizer, criterion, and device for training."""
    learning_rate = config.get('learning_rate', 0.01)
    momentum = config.get('momentum', 0.9)
    weight_decay = config.get('weight_decay', 1e-4)
    epochs = config.get('epochs', 5)
    steps_per_epoch = config.get('steps_per_epoch', 100)  # Will be updated with actual steps
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # optimizer = torch.optim.SGD(
    #     model.parameters(), 
    #     lr=learning_rate, 
    #     momentum=momentum,
    #     weight_decay=weight_decay
    # )
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # OneCycleLR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,  # Warm up for 20% of training
        div_factor=25,  # Initial lr = max_lr/25
        final_div_factor=1e3  # Final lr = initial_lr/1e3
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    return model, optimizer, criterion, scheduler, device


def check_interruption(config):
    """Check if training should be interrupted."""
    interruption_event = config.get('_interruption_event')
    return interruption_event.is_set() if interruption_event else False


def train_single_epoch(model, trainloader, optimizer, criterion, scheduler, device, node_id, epoch, config) -> Metrics:
    """Train model for one epoch and return metrics."""
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    epoch_train_total = 0
    
    for batch_idx, (data, target) in enumerate(trainloader):
        # Check for interruption periodically during training
        if batch_idx % 10 == 0 and check_interruption(config):
            print(f"  Node {node_id} - Training interrupted during epoch {epoch + 1}, batch {batch_idx}")
            return Metrics(0.0, 0.0, 0, interrupted=True, phase="train")
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        epoch_train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        epoch_train_correct += pred.eq(target.view_as(pred)).sum().item()
        epoch_train_total += target.size(0)
    
    train_accuracy = epoch_train_correct / epoch_train_total if epoch_train_total > 0 else 0.0
    avg_train_loss = epoch_train_loss / len(trainloader) if len(trainloader) > 0 else 0.0
    
    return Metrics(avg_train_loss, train_accuracy, epoch_train_total, phase="train")


def validate_single_epoch(model, valloader, criterion, device) -> Metrics:
    """Validate model for one epoch and return metrics."""
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_correct = 0
    epoch_val_total = 0
    
    with torch.no_grad():
        for data, target in valloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            epoch_val_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            epoch_val_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_val_total += target.size(0)
    
    val_accuracy = epoch_val_correct / epoch_val_total if epoch_val_total > 0 else 0.0
    avg_val_loss = epoch_val_loss / len(valloader) if len(valloader) > 0 else 0.0
    
    return Metrics(avg_val_loss, val_accuracy, epoch_val_total, phase="val")


def update_model_with_federated_params(model, federated_params):
    """Update model parameters with federated aggregated parameters."""
    
    federated_arrays = parameters_to_ndarrays(federated_params)
    params_dict = model.state_dict()
    param_keys = list(params_dict.keys())
    
    trainable_idx = 0
    for key in param_keys:
        if params_dict[key].requires_grad and trainable_idx < len(federated_arrays):
            params_dict[key] = torch.from_numpy(federated_arrays[trainable_idx]).to(params_dict[key].device)
            trainable_idx += 1
    
    model.load_state_dict(params_dict)


def perform_centralized_evaluation(model, test_loader, criterion, device, node_id) -> Metrics:
    """Perform centralized evaluation on test data."""
    model.eval()
    epoch_test_loss = 0.0
    epoch_test_correct = 0
    epoch_test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            epoch_test_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            epoch_test_correct += pred.eq(target.view_as(pred)).sum().item()
            epoch_test_total += target.size(0)
    
    test_accuracy = epoch_test_correct / epoch_test_total if epoch_test_total > 0 else 0.0
    test_loss = epoch_test_loss / len(test_loader) if len(test_loader) > 0 else 0.0
    
    print(f"    Node {node_id} - Centralized Test Eval: "
          f"Test Loss: {test_loss:.4f}, Test Acc: {100*test_accuracy:.2f}% "
          f"(Federated model)")
    
    return Metrics(test_loss, test_accuracy, epoch_test_total, phase="test")


def get_model_parameters_as_flower_params(model):
    """Convert model parameters to Flower Parameters format."""
    
    model_params = []
    for param in model.parameters():
        model_params.append(param.data.cpu().numpy())
    
    return ndarrays_to_parameters(model_params)


def create_model():
    """Create a new model instance."""
    return ResNet18(small_resolution=True)


def main():
    """Run the federated learning simulation."""
    print("Serverless PyTorch Federated Learning Simulation")
    print("=" * 60)
    
    # Data loading and preprocessing
    # Training transforms with augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset with different transforms for train and test
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    
    # Create test data loader for centralized evaluation
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Configuration for different experiments
    experiments = [
        {
            "name": "Random Data Distribution",
            "config": {
                "num_nodes": 2,
                "epochs": 30,
                "batch_size": 128,
                "learning_rate": 0.005,
                "momentum": 0.9,
                "weight_decay": 0, # 1e-4,
                "use_async": True,
                "data_split": "random",
            }
        },
    ]
    
    for exp_idx, experiment in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {exp_idx + 1}: {experiment['name']}")
        print(f"{'='*60}")
        
        # Create experiment runner
        runner = FederatedExperimentRunner(experiment['config'])
        
        # Run experiment
        runner.run_experiment(
            model_factory=create_model,
            train_dataset=train_dataset,
            train_fn=train_node,
            test_loader=test_loader,
            num_classes=10   # CIFAR-10 has 10 classes
        )

    print(f"\n{'='*60}")
    print("Simulation completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main() 