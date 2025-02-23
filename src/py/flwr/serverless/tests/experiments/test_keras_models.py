import warnings

warnings.filterwarnings(
    "ignore",
)
import numpy as np
from flwr.server.strategy import FedAvg
from uuid import uuid4
from flwr.serverless.shared_folder.in_memory_folder import InMemoryFolder
from flwr.serverless.keras.example import (
    FederatedLearningTestRun,
)
from flwr.serverless.experiments.model.keras_models import ResNetModelBuilder


# This test is slow on cpu.
def _test_mnist_resnet50_federated_callback_2nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=2,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
        model_builder_fn=ResNetModelBuilder(
            num_classes=10,
            lr=0.001,
            net="ResNet50",
            weights="imagenet",
        ).run,
        replicate_num_channels=True,
        storage_backend=InMemoryFolder(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05
    # assert False  # Uncomment if you want to see the print out of keras training.


# This test fails because of overfitting
def _test_mnist_resnet18_federated_callback_2nodes():
    epochs = 8
    accuracy_standalone, accuracy_federated = FederatedLearningTestRun(
        num_nodes=2,
        epochs=epochs,
        num_rounds=epochs,
        lr=0.001,
        strategy=FedAvg(),
        model_builder_fn=ResNetModelBuilder(
            num_classes=10,
            lr=0.001,
            net="ResNet18",
            # weights="imagenet", # Does not work with ResNet18
        ).run,
        replicate_num_channels=True,
        storage_backend=InMemoryFolder(),
    ).run()
    for i in range(len(accuracy_standalone)):
        assert accuracy_standalone[i] < 1.0 / len(accuracy_standalone) + 0.05

    assert accuracy_federated[0] > accuracy_standalone[0]
    assert accuracy_federated[0] > 1.0 / len(accuracy_standalone) + 0.05
