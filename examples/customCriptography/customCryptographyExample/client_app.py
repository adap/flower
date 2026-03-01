"""authexample: An authenticated Flower / PyTorch app."""
import logging
import os
import time

from flwr.common.logger import log

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
import math  # mettilo in cima al file, insieme agli altri import

MSS = 1460  # TCP payload tipico con MTU 1500 (IPv4)

class FlowerClient(NumPyClient):

    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = get_model(NET, num_classes=10, pretrained=True)
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.process = psutil.Process(os.getpid())
        self.num_cores = psutil.cpu_count(logical=True) or 1


    def _cpu_time(self):
        t = self.process.cpu_times()
        return t.user + t.system

    def _ram_bytes(self):
        return self.process.memory_info().rss

    @staticmethod
    def _bytes_to_mb(value_bytes):
        return value_bytes / (1024 * 1024)

    @staticmethod
    def _extract_round(config):
        for key in ("server_round", "server-round", "current_round", "round"):
            if key in config:
                return config[key]
        return "unknown"

    def fit(self, parameters, config):
        try:
            set_weights(self.net, parameters)
            server_round = self._extract_round(config)

            # misurazione CPU/RAM
            start_cpu = self._cpu_time()
            ram_iniziale_bytes = self._ram_bytes()
            inizio_tempo_reale = time.perf_counter()

            epoche_locali = int(config.get("local_epochs", config.get("local-epochs", self.local_epochs)))
            learning_rate_round = float(config.get("learning_rate", config.get("learning-rate", self.lr)))

            results = train(
                self.net,
                self.trainloader,
                self.valloader,
                epoche_locali,
                learning_rate_round,
                self.device,
            )

            end_cpu = self._cpu_time()
            fine_tempo_reale = time.perf_counter()
            tempo_cpu = end_cpu - start_cpu
            tempo_reale = fine_tempo_reale - inizio_tempo_reale
            core_equivalenti = tempo_cpu / tempo_reale if tempo_reale > 0 else 0.0
            percentuale_cpu = (core_equivalenti / self.num_cores) * 100
            ram_finale_bytes = self._ram_bytes()
            delta_ram_bytes = ram_finale_bytes - ram_iniziale_bytes
            ram_totale_sistema_bytes = psutil.virtual_memory().total
            percentuale_ram_sistema = (ram_finale_bytes / ram_totale_sistema_bytes) * 100
            # cpu_logger.info(f"{cpu_time:.3f}", extra={"pid": os.getpid()})
            weights = get_weights(self.net)

            size_bytes = sum(w.nbytes for w in weights)
            packets = math.ceil(size_bytes / MSS)

            logging.info(
                f"UPLOAD model: {size_bytes} bytes "
                f"(~{size_bytes/(1024*1024):.2f} MB) "
                f"-> ~{packets} pacchetti TCP (MSS={MSS})"
            )

            cpu_line = (
                "CPU per round | pid=%s | round=%s | tempo_cpu=%.3fs | tempo_reale=%.3fs "
                "| core_equivalenti=%.2f | core_logici=%s | percentuale_cpu=%.2f%% | epoche=%s | lr=%.5f"
            )
            log(
                logging.INFO,
                cpu_line,
                os.getpid(),
                server_round,
                tempo_cpu,
                tempo_reale,
                core_equivalenti,
                self.num_cores,
                percentuale_cpu,
                epoche_locali,
                learning_rate_round,
            )

            ram_line = (
                "RAM per round | pid=%s | round=%s | ram_iniziale=%.1fMB | ram_finale=%.1fMB "
                "| delta_ram=%.1fMB | ram_sistema_pct=%.2f%%"
            )
            ram_iniziale_mb = self._bytes_to_mb(ram_iniziale_bytes)
            ram_finale_mb = self._bytes_to_mb(ram_finale_bytes)
            delta_ram_mb = self._bytes_to_mb(delta_ram_bytes)
            log(
                logging.INFO,
                ram_line,
                os.getpid(),
                server_round,
                ram_iniziale_mb,
                ram_finale_mb,
                delta_ram_mb,
                percentuale_ram_sistema,
            )

            return weights, len(self.trainloader.dataset), {
                "tempo_cpu_fit": tempo_cpu,
                "tempo_reale_fit": tempo_reale,
                "core_equivalenti_fit": core_equivalenti,
                "percentuale_cpu_fit": percentuale_cpu,
                "fit_round": server_round,
                "epoche_locali_fit": epoche_locali,
                "learning_rate_fit": learning_rate_round,
                "ram_iniziale_mb_fit": ram_iniziale_mb,
                "ram_finale_mb_fit": ram_finale_mb,
                "delta_ram_mb_fit": delta_ram_mb,
                "percentuale_ram_sistema_fit": percentuale_ram_sistema,
            }
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


    # Crea il client con l'affinity impostata
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
