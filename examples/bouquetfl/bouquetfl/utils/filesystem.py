import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import torch
import yaml
from flwr.common import Code, Status
from flwr.common.typing import Parameters

def load_new_client_state_dict(client_id: int) -> tuple[Status, Parameters]:
    """Load the updated model state dict for a given client after local training. Found in FlowerClient.fit"""
    local_save_path = f"/tmp/params_updated_{client_id}.tp"
    try:
        state_dict_new = torch.load(local_save_path, weights_only=True)
        os.remove(local_save_path)
        status = Status(code=Code.OK, message="Success")

    except FileNotFoundError:
        state_dict_new = None
        status = Status(code=Code.FIT_NOT_IMPLEMENTED, message="Training failed.")
    return status, state_dict_new


def load_client_hardware_config(client_id: int) -> tuple[str, str, int]:
    """Load the hardware configuration for a given client from YAML file. Found in FlowerClient.fit and trainer.py"""
    try:
        with open("./bouquetfl/config/federation_client_hardware.yaml", "r") as f:
            client_config = yaml.safe_load(f)
            gpu = client_config[f"client_{client_id}"]["gpu"]
            cpu = client_config[f"client_{client_id}"]["cpu"]
            ram = client_config[f"client_{client_id}"]["ram_gb"]
        return gpu, cpu, ram
    except FileNotFoundError:
        raise ValueError("Client hardware configuration file not found.")


def save_load_and_training_times(
    client_id: int,
    round: int,
    gpu: str,
    cpu: str,
    data_load_time: float,
    train_time: float,
    num_rounds: int,
    num_clients: int,
) -> None:
    """Save the data load and training times for a given client and round to a pickle file. Found in trainer.py"""
    try:
        df = pd.read_pickle("checkpoints/load_and_training_times.pkl")
    except FileNotFoundError:
        df = pd.DataFrame(
            index=range(0, num_clients),
            columns=["gpu", "cpu"]
            + [f"load_time_{i}" for i in range(1, num_rounds + 1)]
            + [f"train_time_{i}" for i in range(1, num_rounds + 1)],
        )
    df.at[client_id, "gpu"] = gpu
    df.at[client_id, "cpu"] = cpu
    df.at[client_id, f"load_time_{round}"] = data_load_time
    df.at[client_id, f"train_time_{round}"] = train_time
    df.to_pickle("bouquetfl/checkpoints/load_and_training_times.pkl")
