"""FedBABU Client Application.

This module implements the Federated Learning client for the FedBABU (Federated Learning
with Body and Head Update) approach, as described in the paper "FedBABU: Towards Enhanced
Representation Learning in Federated Learning via Backbone Update".

Key Features:
- Implements FedBABU client-side training logic with feature extractor (body) and
  classifier (head) separation
- Supports non-IID data distribution using Dirichlet sampling
- Provides local fine-tuning before evaluation
- Handles model parameter aggregation and distribution
- Configurable hyperparameters through Flower's Context system

The training process follows these steps:
1. During training (fit), only the feature extractor is updated while the classifier
   remains frozen
2. Before evaluation, the entire model is fine-tuned on local data
3. The client maintains synchronization with the global model while preserving local
   adaptations
"""

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, Scalar, Config, Parameters

from fedbabu.task import (
    MobileNetCifar,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

# Default hyperparameters and configuration
NUM_CLASSES = 10  # Number of classes in CIFAR-10 dataset
DEFAULT_ALPHA = 0.1  # Dirichlet concentration parameter for non-IID data distribution
DEFAULT_BATCH_SIZE = 32  # Mini-batch size for training and evaluation
DEFAULT_LEARNING_RATE = 0.1  # Initial learning rate for SGD optimizer
DEFAULT_MOMENTUM = 0.9  # Momentum factor for SGD optimizer
DEFALUT_LOCAL_EPOCHS = 1  # Number of local training epochs per round
DEFAULT_FINETUNE_EPOCHS = 1  # Number of epochs for fine-tuning before evaluation
DEFAULT_NUM_CLASSES_PER_CLIENT = 10  # Number of classes per client in non-IID setting
DEFAULT_TRAIN_TEST_SPLIT_RATIO = 0.2  # Ratio for test set in local data
DEFAULT_SEED = 42  # Random seed for reproducibility

# CIFAR-10 normalization constants
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)


class FedBABUClient(NumPyClient):
    """Flower client implementing FedBABU training strategy.

    This client implements the FedBABU approach where the feature extractor (body) and
    classifier (head) are trained differently:
    - During federation (fit), only the body is updated while the head remains frozen
    - During evaluation, both body and head are fine-tuned on local data
    - The client maintains synchronization with the global model state

    The training process is designed to learn general features in the body through
    federation while allowing local specialization during evaluation.

    Args:
        net (MobileNetCifar): The neural network model with separate body and head
        trainloader (DataLoader): DataLoader for the local training dataset
        valloader (DataLoader): DataLoader for the local validation dataset
        local_epochs (int): Number of local training epochs per federated round
        finetune_epochs (int): Number of epochs for fine-tuning before evaluation
        lr (float): Learning rate for SGD optimizer
        momentum (float): Momentum factor for SGD optimizer
    """

    def __init__(
        self,
        net: MobileNetCifar,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int,
        finetune_epochs: int,
        lr: float,
        momentum: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.finetune_epochs = finetune_epochs
        self.lr = lr
        self.momentum = momentum
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def fit(
        self, parameters: Parameters, config: Config
    ) -> Tuple[Parameters, int, Dict[str, Scalar]]:
        """Train the model's feature extractor on the local dataset.

        This method implements the core FedBABU training logic:
        1. Updates local model with received global parameters
        2. Trains only the feature extractor (body) while keeping classifier (head) frozen
        3. Returns updated parameters and training metrics

        Args:
            parameters (Parameters): Current global model parameters from the server
            config (Config): Training configuration from the server, can include
                           custom parameters for training customization

        Returns:
            Tuple[Parameters, int, Dict[str, Scalar]]: Contains:
                - Updated model parameters after local training
                - Number of training samples used
                - Dictionary with training metrics (e.g., training loss)
        """
        # Update local model with global parameters
        set_weights(self.net, parameters)

        # Perform local training
        train_loss = train(
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            lr=self.lr,
            momentum=self.momentum,
            device=self.device,
        )

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(
        self, parameters: Parameters, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model on local validation data after fine-tuning.

        The evaluation process in FedBABU consists of two steps:
        1. Fine-tune the entire model (both body and head) on local training data
        2. Evaluate the fine-tuned model on local validation data

        This approach allows the model to adapt to local data distributions while
        maintaining the benefits of federated feature learning.

        Args:
            parameters (Parameters): Current global model parameters from the server
            config (Config): Evaluation configuration including:
                           - finetune-epochs: Number of fine-tuning epochs

        Returns:
            Tuple[float, int, Dict[str, Scalar]]: Contains:
                - Validation loss value
                - Number of validation samples
                - Dictionary with metrics (e.g., accuracy)
        """
        # Update local model with global parameters
        set_weights(self.net, parameters)

        # Evaluate model with local fine-tuning
        loss, accuracy = test(
            net=self.net,
            testloader=self.valloader,
            trainloader=self.trainloader,
            device=self.device,
            finetune_epochs=self.finetune_epochs,
            lr=self.lr,
            momentum=self.momentum,
        )

        return loss, len(self.valloader.dataset), {"loss": loss, "accuracy": accuracy}


def client_fn(context: Context) -> NumPyClient:
    """Create and configure a Flower client instance for FedBABU.

    This factory function creates a new client instance for each round of federated
    learning. It handles:
    1. Model initialization
    2. Configuration extraction from context
    3. Data loading and partitioning
    4. Client instantiation with appropriate parameters

    The function supports non-IID data distribution through Dirichlet sampling,
    controlled by the alpha parameter (lower alpha = more non-IID).

    Args:
        context (Context): Contains configuration at different scopes:
            - node_config: Client-specific settings (partition ID, total partitions)
            - run_config: Federation-wide settings including:
                * alpha: Dirichlet concentration parameter
                * local-epochs: Number of local training epochs
                * lr: Learning rate
                * momentum: SGD momentum factor
                * batch-size: Training batch size
                * fraction-fit: Fraction of clients selected per round

    Returns:
        NumPyClient: A configured FedBABUClient instance ready for federation
    """
    # Initialize model
    net = MobileNetCifar()

    # Extract configuration from context
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config.get("local-epochs", DEFALUT_LOCAL_EPOCHS)
    finetune_epochs = context.run_config.get("finetune-epochs", DEFAULT_FINETUNE_EPOCHS)
    lr = context.run_config.get("lr", DEFAULT_LEARNING_RATE)
    momentum = context.run_config.get("momentum", DEFAULT_MOMENTUM)
    batch_size = context.run_config.get("batch-size", DEFAULT_BATCH_SIZE)
    train_test_split_ratio = context.run_config.get(
        "train-test-split-ratio", DEFAULT_TRAIN_TEST_SPLIT_RATIO
    )
    num_classes_per_client = context.run_config.get(
        "num-classes-per-client", DEFAULT_NUM_CLASSES_PER_CLIENT
    )
    seed = context.run_config.get("seed", DEFAULT_SEED)

    # Load and prepare data for this client
    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=num_partitions,
        num_classes_per_client=num_classes_per_client,
        train_test_split_ratio=train_test_split_ratio,
        batch_size=batch_size,
        seed=seed,
    )

    # Create and return client instance
    return FedBABUClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        local_epochs=local_epochs,
        finetune_epochs=finetune_epochs,
        lr=lr,
        momentum=momentum,
    ).to_client()


# Create Flower client application
app = ClientApp(client_fn)
