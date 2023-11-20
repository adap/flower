import flwr as fl
from typing import Dict, Tuple, List
from flwr.common.typing import Scalar
import torch

from collections import OrderedDict
from client import test2, generate, load_data_mnist

from models_mnist import Net

# _, val_loader = load_data_mnist()


def main():
    fl.server.start_server(
        strategy=fl.server.strategy.FedAvg(on_fit_config_fn=get_on_fit_fn()),
        # strategy=fl.server.strategy.FedAvg(evaluate_fn=get_evaluate_fn(val_loader)),
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
    )


def set_params(model: torch.nn.ModuleList, params: List[fl.common.NDArrays]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_on_fit_fn():
    def fit_config(server_round):
        return {"server_round": server_round}

    return fit_config


def get_evaluate_fn(
    testset,
):
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        """Use the entire test set for evaluation."""

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        model.to(device)
        set_params(model, parameters)
        # testloader = DataLoader(testset, batch_size=50)
        testloader = testset
        loss = test2(model, testloader, gen=True, rnd=server_round)

        return loss, {}

    return evaluate


if __name__ == "__main__":
    main()
