"""opacus: Training with Sample-Level Differential Privacy using Opacus Privacy Engine."""

import logging
import warnings

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from opacus import PrivacyEngine
from opacus_fl.task import Net, get_weights, load_data, set_weights, test, train

warnings.filterwarnings("ignore", category=UserWarning)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        train_loader,
        test_loader,
        target_delta,
        noise_multiplier,
        max_grad_norm,
    ) -> None:
        super().__init__()
        self.model = Net()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        model = self.model
        set_weights(model, parameters)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        privacy_engine = PrivacyEngine(secure_mode=False)
        (
            model,
            optimizer,
            self.train_loader,
        ) = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=self.train_loader,
            noise_multiplier=self.noise_multiplier,
            max_grad_norm=self.max_grad_norm,
        )

        epsilon = train(
            model,
            self.train_loader,
            privacy_engine,
            optimizer,
            self.target_delta,
            device=self.device,
        )

        if epsilon is not None:
            print(f"Epsilon value for delta={self.target_delta} is {epsilon:.2f}")
        else:
            print("Epsilon value not available.")

        return (get_weights(model), len(self.train_loader.dataset), {})

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        return loss, len(self.test_loader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    noise_multiplier = 1.0 if partition_id % 2 == 0 else 1.5

    train_loader, test_loader = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    return FlowerClient(
        train_loader,
        test_loader,
        context.run_config["target-delta"],
        noise_multiplier,
        context.run_config["max-grad-norm"],
    ).to_client()


app = ClientApp(client_fn=client_fn)
