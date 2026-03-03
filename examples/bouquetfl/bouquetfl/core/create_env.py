import json
import os
import subprocess
import tempfile

from flwr.app import Context, Message
from flwr.common import Status
from flwr.common.typing import Parameters
from bouquetfl.utils import power_clock_tools as pct
from bouquetfl.utils.filesystem import (load_client_hardware_config,
                                        load_new_client_state_dict)
from bouquetfl.utils.localinfo import get_local_gpu_cores


def run_training_process_in_env(
    msg: Message, context: Context, mode: str,
) -> tuple[Status, Parameters]:

    gpu_name, cpu_name, ram = load_client_hardware_config(context.node_config["partition-id"])
    print(
        f"{"\033[31m"}Client {context.node_config["partition-id"]} hardware{"\033[0m"}: GPU={gpu_name}, CPU={cpu_name}, RAM={ram}GB"
    )

    # Determine GPU core percentage for MPS
    current_cores = get_local_gpu_cores()
    target_cores = pct.get_gpu_info(gpu_name)["cuda cores"]
    env = os.environ.copy()
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(min(100, 100 * target_cores / current_cores))
    env["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-mps"


    # We run trainer.py as a separate process with systemd-run using a set CUDA_MPS_ACTIVE_THREAD_PERCENTAGE.
    # Anything else (CPU throttling, RAM limiting, GPU memory and clock limiting) could be done without a separate process.

    # We take advantage of systemd-run to limit the RAM usage of the process.

    cfg = dict(context.run_config)
    cfg.update(msg.content["config"])
    cfg["mode"] = mode


    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        cfg_path = f.name

    child = subprocess.Popen(
        [
            "systemd-run",
            "--user",
            "--scope",
            "-p",
            f"MemoryMax={ram}G",
            "poetry ",  # <--- Change this to "python3", "poetry", "pyenv" or corresponding if you don't have uv installed
            "run",
            "./bouquetfl/core/trainer.py",
            "--client_id",
            f"{context.node_config['partition-id']}",
            "--config",
            f"{cfg_path}",
        ],
        stdin=subprocess.PIPE,
        #stdout=subprocess.DEVNULL,    # Comment this line out for verbose from the training process
        #stderr=subprocess.DEVNULL,    # Comment this line out for verbose from the training process
        text=True,
        env=env,
    )

    # Wait for the subprocess to finish before spawning the next one
    child.wait()
    pct.reset_all_limits()

    # Get new stored model parameters and return to server

    status, state_dict_updated = load_new_client_state_dict(
        context.node_config["partition-id"]
    )
    return status, state_dict_updated
