from collections import OrderedDict
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar


from hydra.utils import instantiate

import torch
import flwr as fl

from model import train, test


class FlowerClient(fl.client.NumPyClient):
    """A standard FlowerClient."""

    def __init__(self, trainloader, vallodaer, model_cfg) -> None:
        super().__init__()

        self.trainloader = trainloader
        self.valloader = vallodaer

        # For further flexibility, we don't hardcode the type of model we use in
        # federation. Here we are instantiating the object defined in `conf/model/net.yaml`
        # (unless you changed the default) and by then `num_classes` would already be auto-resolved
        # to `num_classes=10` (since this was known right from the moment you launched the experiment)
        self.model = instantiate(model_cfg)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config: Dict[str, Scalar]):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)

        lr = config["lr"]
        momentum = config["momentum"]
        epochs = config["local_epochs"]

        # You could also set this optimiser from a config file. That would make it
        # easy to run experiments considering different optimisers and set one or another
        # directly from the command line (you can use as inspiration what we did for adding
        # support for FedAvg and FedAdam strategies)
        optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

        # do local training
        # similarly, you can set this via a config. For example, imagine you have different
        # experiments using wildly different training protocols (e.g. vision, speech). You can
        # toggle between different training functions directly from the config without having
        # to clutter your code with if/else statements all over the place :)
        train(self.model, self.trainloader, optim, epochs, self.device)

        return self.get_parameters({}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        self.set_parameters(parameters)

        loss, accuracy = test(self.model, self.valloader, self.device)

        return float(loss), len(self.valloader), {"accuracy": accuracy}


def generate_client_fn(trainloaders, valloaders, model_cfg):
    """Return a function to construct a FlowerClient."""

    def client_fn(cid: str):
        return FlowerClient(
            trainloader=trainloaders[int(cid)],
            vallodaer=valloaders[int(cid)],
            model_cfg=model_cfg,
        ).to_client()

    return client_fn
