import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import argparse
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")


class CifarClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        trainset: torchvision.datasets,
        testset: torchvision.datasets,
        validation_split: int = 0.1,
    ):
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.validation_split = validation_split

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.set_parameters(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]

        n_valset = int(len(self.trainset) * self.validation_split)

        valset = torch.utils.data.Subset(self.trainset, range(0, n_valset))
        trainset = torch.utils.data.Subset(
            self.trainset, range(n_valset, len(self.trainset))
        )

        trainLoader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        valLoader = DataLoader(valset, batch_size=batch_size)

        results = utils.train(self.model, trainLoader, valLoader, epochs)

        parameters_prime = [
            val.cpu().numpy() for _, val in self.model.state_dict().items()
        ]
        num_examples_train = len(trainset)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""
        # Update local model parameters
        self.set_parameters(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        testloader = DataLoader(self.testset, batch_size=steps)

        loss, accuracy = utils.test(self.model, testloader)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}


def client_dry_run():
    """Weak tests to check whether all client methods are working as expected."""
    model = utils.load_efficientnet(classes=10)
    trainset, testset = utils.load_partition(0)
    trainset = torch.utils.data.Subset(trainset, range(10))
    testset = torch.utils.data.Subset(testset, range(10))
    client = CifarClient(model, trainset, testset)
    client.fit(
        [val.cpu().numpy() for _, val in model.state_dict().items()],
        {"batch_size": 32, "local_epochs": 1},
    )

    client.evaluate(
        [val.cpu().numpy() for _, val in model.state_dict().items()], {"val_steps": 32}
    )

    print("Dry Run Successful")


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 10), required=True)
    args = parser.parse_args()

    # Load PyTorch model
    model = utils.load_efficientnet(classes=10)

    # Load a subset of CIFAR-10 to simulate the local data partition
    trainset, testset = utils.load_partition(args.partition)

    if True:
        trainset = torch.utils.data.Subset(trainset, range(10))
        testset = torch.utils.data.Subset(testset, range(10))

    # Start Flower client
    client = CifarClient(model, trainset, testset)

    fl.client.start_numpy_client("0.0.0.0:8080", client=client)


if __name__ == '__main__':
    main()
