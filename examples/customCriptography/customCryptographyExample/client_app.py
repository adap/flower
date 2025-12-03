"""authexample: An authenticated Flower / PyTorch app."""
import logging
import os

import numpy as np
import psutil
import torch


from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.crypto.config_cripto import NET

from .task import (
    get_model,
    get_weights,
    load_data_from_disk,
    set_weights,
    test,
    train,
)

# Logger CPU
# cpu_logger = logging.getLogger("cpu_logger")
# cpu_logger.setLevel(logging.INFO)
# file_handler = logging.FileHandler("cpu_usage.log", mode="a")
# formatter = logging.Formatter("%(asctime)s [INFO] Client %(pid)d CPU time durante fit: %(message)s s")
# file_handler.setFormatter(formatter)
# cpu_logger.addHandler(file_handler)


# -------------------------------------------------------------------
# CLIENT CON ASSEGNAZIONE AUTOMATICA DEI CORE
# -------------------------------------------------------------------
class FlowerClient(NumPyClient):

    def __init__(self, trainloader, valloader, local_epochs, learning_rate, core_list):
        self.net = get_model(NET, num_classes=10, pretrained=False)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.process = psutil.Process(os.getpid())


    def _cpu_time(self):
        t = self.process.cpu_times()
        return t.user + t.system
    def fit(self, parameters, config):
        print(f"[CLIENT {os.getpid()}] fit() STARTED", flush=True)
        try:
            set_weights(self.net, parameters)

            # misurazione CPU
            start_cpu = self._cpu_time()

            results = train(
                self.net,
                self.trainloader,
                self.valloader,
                self.local_epochs,
                self.lr,
                self.device,
            )

            end_cpu = self._cpu_time()
            cpu_time = end_cpu - start_cpu
            # cpu_logger.info(f"{cpu_time:.3f}", extra={"pid": os.getpid()})

            return get_weights(self.net), len(self.trainloader.dataset), {"cpu_fit": cpu_time}
        except Exception:
            logging.exception("ERRORE in fit() sul client, il client sta crashando!")
            raise

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


# -------------------------------------------------------------------
# CREAZIONE CLIENT - ASSEGNAZIONE CORES PER CLIENT
# -------------------------------------------------------------------
def client_fn(context: Context):

    dataset_path = context.node_config["dataset-path"]

    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data_from_disk(dataset_path, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # -----------------------------
    # Assegnazione dinamica core
    # -----------------------------
    CORES_PER_CLIENT = 3
    client_id = context.node_id  # ID univoco del client

    start = client_id * CORES_PER_CLIENT
    end = start + CORES_PER_CLIENT
    core_list = list(range(start, end))

    print(f"[SuperNode] Client {client_id} -> core assegnati: {core_list}")

    # Crea il client con l'affinity impostata
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate, core_list).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
