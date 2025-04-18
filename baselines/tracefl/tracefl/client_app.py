import copy
import json

import toml
import torch
from omegaconf import OmegaConf

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tracefl.dataset import (
    get_clients_server_data,
)
from tracefl.fls import FLSimulation
from tracefl.models import (
    get_parameters,
    initialize_model,
    set_parameters,
    train_neural_network,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        net,
        trainloader,
        local_epochs,
        partition_id,
        cfg,
        ds_dict,
        arch,
        model_dict,
        client2data,
    ):
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.cid = partition_id
        self.is_faulty = False
        self.cfg = cfg
        self.ds_dict = ds_dict
        self.arch = arch
        self.model_dict = model_dict
        self.client2data = client2data

    def fit(self, parameters, config):

        set_parameters(self.net, parameters)

        tconfig = {
            "arch": self.arch,
            "model_dict": self.model_dict,
            "train_data": self.ds_dict["client2data"][str(self.cid)],
            "batch_size": self.cfg.tool.tracefl.client.batch_size,
            "lr": self.cfg.tool.tracefl.client.lr,
            "epochs": self.cfg.tool.tracefl.client.epochs,
            "device": self.device,
        }
        train_result = train_neural_network(tconfig)
        label_counts = {}
        labels = self.ds_dict["client2data"][str(self.cid)]["label"]
        for label in labels:
            label = int(label)
            label_counts[label] = label_counts.get(label, 0) + 1

        return (
            get_parameters(self.net),
            len(self.ds_dict["client2data"][str(self.cid)]),
            {
                "train_loss": train_result["train_loss"],
                "train_accuracy": train_result["train_accuracy"],
                "client_message": "Hello from Client!",
                "cid": self.cid,
                "class_distribution": json.dumps(label_counts),
            },
        )

    def evaluate(self, parameters, config):
        pass


def client_fn(context: Context):

    config = toml.load("./tracefl/resnet.toml")
    cfg = OmegaConf.create(config)
    partition_id = int(context.node_config["partition-id"])
    ds_dict = get_clients_server_data(cfg)

    client_train_data = ds_dict["client2data"].get(str(partition_id))

    sim = FLSimulation(copy.deepcopy(cfg), 0.0, 0.0, 0.0, 0.0)
    sim.set_clients_data(ds_dict["client2data"])
    sim.set_strategy()
    model_dict = initialize_model(cfg.tool.tracefl.model.name, cfg.tool.tracefl.dataset)
    local_epochs = context.run_config["local-epochs"]
    return FlowerClient(
        model_dict["model"],
        client_train_data,
        local_epochs,
        partition_id,
        cfg,
        ds_dict,
        cfg.tool.tracefl.model.arch,
        model_dict,
        ds_dict["client2data"],
    ).to_client()


app = ClientApp(client_fn=client_fn)
