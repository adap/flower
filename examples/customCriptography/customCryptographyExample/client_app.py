"""authexample: An authenticated Flower / PyTorch app."""
import logging
import os

import numpy as np
import psutil
import torch
# ##Per resnet 34ù
# ##Per resnet 34
torch.set_num_threads(3)
# ##Per resnet 18
# #torch.set_num_threads(6)
torch.set_num_interop_threads(1)
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
# Ora puoi fare un import assoluto
from flwr.common.crypto.config_cripto import NET

from .task import (
    get_model,
    get_weights,
    load_data_from_disk,
    set_weights,
    test,
    train,
)



# Logger dedicato solo per CPU
cpu_logger = logging.getLogger("cpu_logger")
cpu_logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("cpu_usage.log", mode="a")
formatter = logging.Formatter("%(asctime)s [INFO] Client %(pid)d CPU time durante fit: %(message)s s")
file_handler.setFormatter(formatter)
cpu_logger.addHandler(file_handler)
# Define Flower Client
class FlowerClient(NumPyClient):
    import logging
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = get_model(NET, num_classes=10, pretrained=False)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.proc = psutil.Process(os.getpid())

    def _cpu_time(self):
        t = self.proc.cpu_times()
        return t.user + t.system
    def fit(self, parameters, config):
        print(f"[CLIENT {os.getpid()}] fit() STARTED", flush=True)
        try:
            set_weights(self.net, parameters)

            # inizio misurazione CPU
            start_cpu = self._cpu_time()

            results = train(
                self.net,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
            )

            # fine misurazione CPU
            end_cpu = self._cpu_time()
            cpu_time = end_cpu - start_cpu
            cpu_logger.info(f"{cpu_time:.3f}", extra={"pid": os.getpid()})

            return get_weights(self.net), len(self.trainloader.dataset), {"cpu_fit": cpu_time}
        except Exception:
            logging.exception("ERRORE in fit() sul client, il client sta crashando!")
            raise

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to get the path to the dataset the SuperNode running
    # this ClientApp has access to
    dataset_path = context.node_config["dataset-path"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data_from_disk(dataset_path, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
