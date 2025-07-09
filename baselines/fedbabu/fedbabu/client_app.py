"""Federated Learning Client Application.

This module implements client-side training for both FedBABU (Federated Learning
with Body and Head Update) and FedAvg approaches, as described in the paper
"FedBABU: Towards Enhanced Representation for Federated Image Classification".

Key Features:
- Implements both FedBABU and FedAvg training strategies
- Uses feature extractor (body) and classifier (head) separation for FedBABU
- Supports non-IID data distribution using class-based partitioning
- Provides local fine-tuning before evaluation for FedBABU
- Handles model parameter aggregation and distribution
- Configurable hyperparameters through Flower's Context system

The training process varies by algorithm:
FedBABU:
1. During training (fit), only the feature extractor is updated while the classifier
   remains frozen
2. Before evaluation, the entire model is fine-tuned on local data
3. The client maintains synchronization with the global model while preserving local
   adaptations

FedAvg:
1. During training (fit), the entire model is updated
2. No special treatment during evaluation
3. Full model synchronization with the global state
"""

from typing import Dict, Tuple

import torch
from fedbabu.task import FourConvNet, get_weights, load_data, set_weights, test, train
from flwr.client import ClientApp, NumPyClient
from flwr.common import Config, Context, Parameters, Scalar
from torch.utils.data import DataLoader

# Default hyperparameters and configuration
NUM_CLASSES = 10  # Number of classes in CIFAR-10 dataset
DEFAULT_BATCH_SIZE = 32  # Mini-batch size for training and evaluation
DEFAULT_LEARNING_RATE = 0.1  # Initial learning rate for SGD optimizer
DEFAULT_MOMENTUM = 0.9  # Momentum factor for SGD optimizer
DEFALUT_LOCAL_EPOCHS = 4  # Number of local training epochs per round
DEFAULT_FINETUNE_EPOCHS = 1  # Number of epochs for fine-tuning before evaluation
DEFAULT_NUM_CLASSES_PER_CLIENT = 2  # Number of classes per client in non-IID setting
DEFAULT_TRAIN_TEST_SPLIT_RATIO = 0.2  # Ratio for test set in local data
DEFAULT_SEED = 42  # Random seed for reproducibility

# CIFAR-10 normalization constants
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)


class FlowerClient(NumPyClient):
    """Flower client implementing FedBABU and FedAvg training strategies.

    This client supports two training strategies:

    FedBABU:
    - During federation (fit), only the body is updated while the head remains frozen
    - During evaluation, both body and head are fine-tuned on local data
    - The client maintains synchronization with the global model state

    FedAvg:
    - During federation (fit), the entire model is updated
    - No special treatment during evaluation
    - Full model synchronization across clients

    The training process is designed to either learn general features through
    federation (FedBABU) or perform standard federated averaging (FedAvg).

    Args:
        algorithm (str): Training strategy to use ("fedbabu" or "fedavg")
        net (FourConvNet): The neural network model with separate body and head
        trainloader (DataLoader): DataLoader for the local training dataset
        valloader (DataLoader): DataLoader for the local validation dataset
        local_epochs (int): Number of local training epochs per federated round
        finetune_epochs (int): Number of epochs for fine-tuning before evaluation
        lr (float): Learning rate for SGD optimizer
        momentum (float): Momentum factor for SGD optimizer

    Raises:
        AssertionError: If algorithm is not "fedbabu" or "fedavg"
    """

    def __init__(
        self,
        algorithm: str,
        net: FourConvNet,
        trainloader: DataLoader,
        valloader: DataLoader,
        local_epochs: int,
        finetune_epochs: int,
        lr: float,
        momentum: float,
    ) -> None:
        assert algorithm in ["fedbabu", "fedavg"], "Unsupported algorithm"
        self.algorithm = algorithm
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
        """Train the model on the local dataset.

        This method implements two training strategies based on self.algorithm:

        FedBABU:
        1. Updates local model with received global parameters
        2. Trains only the feature extractor (body) while keeping classifier frozen
        3. Returns updated parameters and training metrics

        FedAvg:
        1. Updates local model with received global parameters
        2. Trains the entire model without freezing any parts
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
            algorithm=self.algorithm,
            net=self.net,
            trainloader=self.trainloader,
            epochs=self.local_epochs,
            lr=config.get("lr", self.lr),
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
        """Evaluate the model on local validation data.

        The evaluation process varies by algorithm:

        FedBABU:
        1. Fine-tune the entire model (both body and head) on local training data
        2. Evaluate the fine-tuned model on local validation data
        This allows the model to adapt to local data distributions while
        maintaining the benefits of federated feature learning

        FedAvg:
        1. Use the global model parameters as-is
        2. Evaluate directly on local validation data
        This maintains consistent model behavior across all clients

        Args:
            parameters (Parameters): Current global model parameters from the server
            config (Config): Evaluation configuration including:
                           - finetune-epochs: Number of fine-tuning epochs (FedBABU only)

        Returns:
            Tuple[float, int, Dict[str, Scalar]]: Contains:
                - Validation loss value
                - Number of validation samples
                - Dictionary with metrics (loss and accuracy)
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
            lr=config.get("lr", self.lr),
            momentum=self.momentum,
        )

        return loss, len(self.valloader.dataset), {"loss": loss, "accuracy": accuracy}


def client_fn(context: Context) -> NumPyClient:
    """Create and configure a Flower client instance.

    This factory function creates a new client instance for each round of federated
    learning. It handles:
    1. Model initialization
    2. Algorithm selection (FedBABU or FedAvg)
    3. Configuration extraction from context
    4. Data loading and partitioning
    5. Client instantiation with appropriate parameters

    The function supports non-IID data distribution through class-based partitioning,
    where each client receives data from a limited number of classes to simulate
    realistic federated learning scenarios.

    Args:
        context (Context): Contains configuration at different scopes:
            - node_config: Client-specific settings:
                * partition-id: ID of this client's data partition
                * num-partitions: Total number of data partitions
            - run_config: Federation-wide settings including:
                * algorithm: Training strategy ("fedbabu" or "fedavg")
                * local-epochs: Number of local training epochs
                * finetune-epochs: Number of fine-tuning epochs
                * lr: Learning rate
                * momentum: SGD momentum factor
                * batch-size: Training batch size
                * num-classes-per-client: Number of classes per client
                * train-test-split-ratio: Ratio for local validation set
                * seed: Random seed for reproducibility

    Returns:
        NumPyClient: A configured FlowerClient instance ready for federation
    """
    # Initialize model
    net = FourConvNet()

    # Extract configuration from context
    algorithm = context.run_config["algorithm"]
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
    return FlowerClient(
        algorithm=algorithm,
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
