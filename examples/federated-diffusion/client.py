import json
import os
import pickle
from collections import OrderedDict

import torch
from centralized import get_model, train, validate
from data import load_datasets

import flwr as fl


def get_parameters(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def save_personalization_weight(cid, model, personalization_layers, path):
    weights = get_parameters(model)
    # save weight
    personalized_weight = weights[len(weights) - personalization_layers :]
    os.makedirs(path, exist_ok=True)
    with open(f"{path}/Per_{cid}.pickle", "wb") as file_weight:
        pickle.dump(personalized_weight, file_weight)
    file_weight.close()


def load_personalization_weight(cid, model, personalization_layers, path):
    weights = get_parameters(model)
    with open(f"{path}/Per_{cid}.pickle", "rb") as file_weight:
        personalized_weight = pickle.load(file_weight)
        file_weight.close()
    weights[len(weights) - personalization_layers :] = personalized_weight

    # set new weight to the model
    set_parameters(model, weights)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, cid, device, args):
        self.model = model
        self.trainloader = trainloader
        self.cid = cid
        self.device = device
        self.args = args

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # Update local model parameters
        set_parameters(self.model, parameters)

        # Read values from config
        server_round = config["server_round"]

        cpu = False
        if str(self.device) == "cpu":
            cpu = True

        # Update local model parameters
        if int(server_round) > 1 and self.args.personalized:
            load_personalization_weight(
                self.cid,
                self.model,
                self.args.personalization_layers,
                self.args.personalization_path,
            )

        train(
            self.args,
            self.model,
            self.trainloader,
            self.cid,
            server_round,
            cpu,
        )
        if self.args.personalized:
            save_personalization_weight(
                self.cid,
                self.model,
                self.args.personalization_layers,
                self.args.personalization_path,
            )

        return get_parameters(self.model), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # Update local model parameters
        set_parameters(self.model, parameters)

        server_round = config["server_round"]

        precision, recall, num_examples = validate(
            self.args, self.model, self.cid, self.device
        )
        results = {
            "precision": precision,
            "recall": recall,
            "cid": self.cid,
            "server_round": server_round,
        }
        os.makedirs(self.args.log_path, exist_ok=True)
        json.dump(results, open(f"{self.args.log_path}/logs.json", "a"))
        results.pop("server_round")

        return (
            1.0,
            num_examples,
            results,
        )


def gen_client_fn(args):
    trainloaders = load_datasets(args, args.iid)

    def client_fn(cid):
        """Create a Flower client representing a single organization."""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Load model
        model = get_model().to(device)

        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            model,
            trainloader,
            cid,
            device,
            args,
        )

    return client_fn
