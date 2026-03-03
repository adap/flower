import argparse
import json
import os
import time
import timeit

import pyarrow as pa
import torch

from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (load_client_hardware_config,
                                        save_load_and_training_times)
from bouquetfl import task

os.environ["HF_DATASETS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pa.set_cpu_count(1)

# Arguments passed by client.py

parser = argparse.ArgumentParser(
    description="Train a client-specific model with specific hardware settings."
)
args_list = [
    ("--client_id", {"type": int, "default": 0, "help": "Client ID."}),
    (
        "--config",
        {"type": str, "default": "", "help": "Path to config from pyproject.toml"},
    ),
]
for arg, kwargs in args_list:
    parser.add_argument(arg, **kwargs)
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

####################################
############# Training #############
####################################


def train_model():
    gpu, cpu, _ = load_client_hardware_config(args.client_id)
    # Load model and apply global parameters
    model = task.get_model()
    try:
        state_dict_global = torch.load(
            f"/tmp/global_params_round_{cfg['server-round']}.pt",
            weights_only=True,
        )
    except FileNotFoundError:
        state_dict_global = {}
        state_dict_global = task.get_initial_state_dict()

    # Set hardware limits (Ram limit was set in the subprocess environement)
    pct.set_physical_gpu_limits(gpu)
    num_cpu_cores = pct.set_cpu_limit(cpu)
    # Give some time for the limits to take effect
    time.sleep(0.5)

    # Load data (on CPU)
    start_data_load_time = timeit.default_timer()
    if cfg["mode"] == "simulation":
        trainloader = task.load_data(
            args.client_id,
            num_clients=cfg["num-partitions"],
            num_workers=num_cpu_cores,
            batch_size=cfg["batch-size"],
        )
    elif cfg["mode"] == "real":
        trainloader = task.load_data_real(
            args.client_id,
            num_workers=num_cpu_cores,
            batch_size=cfg["batch-size"],
        )
    data_load_time = timeit.default_timer() - start_data_load_time

    # Train model (on GPU)
    start_train_time = timeit.default_timer()
    model.load_state_dict(state_dict_global)

    try:
        task.train(
            model=model,
            trainloader=trainloader,
            epochs=cfg["local-epochs"],
            device=("cuda" if torch.cuda.is_available() and gpu != "None" else "cpu"),
            lr=cfg["learning-rate"],
        )
        train_time = timeit.default_timer() - start_train_time
        # Save updated model parameters
        torch.save(model.state_dict(), f"/tmp/params_updated_{args.client_id}.tp")

    except torch.OutOfMemoryError:
        # print(f"Client {args.client_id} has encountered an out-of-memory error.")
        train_time = -1.0
        try:
            os.remove(f"/tmp/params_updated_{args.client_id}.tp")
        except FileNotFoundError:
            pass


    # Save load and training times
    save_load_and_training_times(
        client_id=args.client_id,
        round=cfg["server-round"],
        gpu=gpu,
        cpu=cpu,
        data_load_time=data_load_time,
        train_time=train_time,
        num_rounds=cfg["num-server-rounds"],
        num_clients=cfg["num-partitions"],
    )


train_model()
