import shutil
import subprocess

import keyring
import pandas as pd
import torch
import yaml
# The alt.file source is required when running on Ubuntu-server without a GUI, else just use keyring.PlaintextKeyring
from keyrings.alt.file import PlaintextKeyring

keyring.set_keyring(PlaintextKeyring())

service = "power_clock_tools_service"
username = "local_user"

password = keyring.get_password(service, username)
if password is None:
    password = input("Enter sudo password: ")
    keyring.set_password(service, username, password)
    print("Password saved securely.")

#####################################
############# Auxiliary #############
#####################################


def run(cmd):
    subprocess.run(
        ["sudo", "-S"] + cmd,
        input=password + "\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


def require_sudo():
    if shutil.which("sudo") is None:
        print("sudo not found. On some systems you must run as root.")
    return []


#####################################
############# GPU tools #############
#####################################


def set_gpu_memory_limit(value: int, gpu_index: int):
    "Sets total available memory of the process to <value> GB"
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_fraction = min(1, float(value * 1024**3) / total_memory)
    torch.cuda.memory.set_per_process_memory_fraction(memory_fraction, gpu_index)


def reset_gpu_memory_limit(gpu_index: int):
    torch.cuda.memory.set_per_process_memory_fraction(1.0, gpu_index)


def lock_gpu_clocks(gpu_index: int, min_mhz: int, max_mhz: int):
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--lock-gpu-clocks={min_mhz},{max_mhz}",
    ]
    run(cmd)


def reset_gpu_clocks(gpu_index: int):
    # Requires sudo; only supported on Volta+.
    cmd = ["nvidia-smi", "-i", str(gpu_index), "--reset-gpu-clocks"]
    run(cmd)


def lock_gpu_memory_clocks(gpu_index: int, min_mhz: int, max_mhz: int):
    # Requires sudo; only supported on Volta+.
    cmd = [
        "nvidia-smi",
        "-i",
        str(gpu_index),
        f"--lock-memory-clocks={min_mhz},{max_mhz}",
    ]
    run(cmd)


def reset_gpu_memory_clocks(gpu_index: int):
    # Requires sudo; only supported on Volta+.
    cmd = ["nvidia-smi", "-i", str(gpu_index), "--reset-memory-clocks"]
    run(cmd)


def get_gpu_info(gpu_name: str):
    gpu_info = None
    with open("./bouquetfl/hardwareconf/gpus.csv") as file:
        gpus = pd.read_csv(file, header=None).to_numpy()
    for gpu in gpus:
        if gpu[0] == gpu_name:
            gpu_info = {
                "name": gpu[0],
                "memory": float(gpu[1]),
                "memory type": gpu[2],
                "memory bandwidth": float(gpu[3]),
                "clock speed": float(gpu[4]),
                "memory speed": float(gpu[5]),
                "cuda cores": int(gpu[6]),
            }

    if not gpu_info:
        raise ValueError(f"GPU {gpu_name} not found in database.")

    return gpu_info


def set_physical_gpu_limits(gpu_name: str):
    gpu_info = get_gpu_info(gpu_name)
    
    with open("./bouquetfl/config/local_hardware.yaml", "r") as stats_file:
        current_hardware_info = yaml.safe_load(stats_file)
    if not gpu_info:
        raise ValueError(f"GPU {gpu_name} not found in database.")

    if gpu_info["memory"] > int(current_hardware_info["gpu_memory"]):
        raise ValueError(
            f"GPU {gpu_name} has more memory ({gpu_info['memory']} GB) than the current GPU ({current_hardware_info['gpu_memory']} GB)."
        )
    if gpu_info["clock speed"] > int(current_hardware_info["gpu_clock_speed"]):
        raise ValueError(
            f"GPU {gpu_name} has a higher clock speed ({gpu_info['clock speed']} MHz) than the current GPU ({current_hardware_info['gpu_clock_speed']} MHz)."
        )
    if gpu_info["memory speed"] > int(current_hardware_info["gpu_memory_speed"]):
        raise ValueError(
            f"GPU {gpu_name} has a higher memory speed ({gpu_info['memory speed']} MHz) than the current GPU ({current_hardware_info['gpu_memory_speed']} MHz)."
        )

    set_gpu_memory_limit(gpu_info["memory"], 0)

    lock_gpu_clocks(0, int(gpu_info["clock speed"]), int(gpu_info["clock speed"]))

    lock_gpu_memory_clocks(
        0, int(gpu_info["memory speed"]), int(gpu_info["memory speed"])
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(f"GPU memory speed: {gpu_info['memory speed']} MHz; GPU memory limit: {gpu_info['memory']} GB;  GPU clock speed: {gpu_info['clock speed']} MHz"
    )


#####################################
############# CPU tools #############
#####################################


def get_cpu_info(cpu_name: str):
    cpu_info = None
    with open("./bouquetfl/hardwareconf/cpus.csv") as file:
        cpus = pd.read_csv(file, header=None).to_numpy()
    for cpu in cpus:
        if cpu[0] == cpu_name:
            if len(cpu[1].split(" ")) > 1:
                num_cores = cpu[1].split(" ")[0]
            else:
                num_cores = cpu[1]
            base_clock = cpu[2].split(" ")[0]
            turbo_clock = base_clock
            if len(cpu[2].split(" ")) > 2:
                turbo_clock = cpu[2].split(" ")[2]
            cpu_info = {
                "name": cpu[0],
                "cores": int(num_cores),
                "base clock": 1000 * float(base_clock),  # GHz to MHz
                "turbo clock": 1000 * float(turbo_clock),  # GHz to MHz
            }

    if not cpu_info:
        raise ValueError(f"CPU {cpu_name} not found in database.")

    return cpu_info


def set_cpu_limit(cpu_name: str):
    cpu_info = get_cpu_info(cpu_name)
    with open("./bouquetfl/config/local_hardware.yaml", "r") as stats_file:
        current_hardware_info = yaml.safe_load(stats_file)
    if not cpu_info:
        raise ValueError(f"CPU {cpu_name} not found in database.")

    if cpu_info["cores"] > int(current_hardware_info["cpu_cores"]):
        raise ValueError(
            f"CPU {cpu_name} has more cores ({cpu_info['cores']}) than the current CPU ({current_hardware_info['cpu_cores']})."
        )
    if cpu_info["base clock"] > int(current_hardware_info["cpu_clock_speed"]):
        print(
            f"CPU {cpu_name} has a higher clock speed ({cpu_info['base clock']} MHz) than the current CPU ({current_hardware_info['cpu_clock_speed']} MHz)."
        )

    cmd = [
        "cpupower",
        "frequency-set",
        "-u",
        f"{cpu_info['base clock']}MHz",
    ]
    run(cmd)
    print(f"Set CPU clock speed to {cpu_info['base clock']} MHz")
    return cpu_info["cores"]


def reset_cpu_limit():
    # Resets CPU governor to performance
    cmd = [
        "cpupower",
        "frequency-set",
        "-g",
        "performance",
    ]
    run(cmd)


def reset_all_limits():
    reset_cpu_limit()
    reset_gpu_memory_limit(0)
    reset_gpu_clocks(0)
    reset_gpu_memory_clocks(0)
