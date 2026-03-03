import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import yaml
from bouquetfl.utils.localinfo import profile_local_hardware


# Steam Hardware Survey based sampling for GPUs and CPUs: hardware stats for Windows Computers (94.84% of total)
# Source: https://store.steampowered.com/hwsurvey/processormfg/   (October 2025)
# To generate samples, we first check the current hardware capabilities of the physical machine
# Then, we sample hardware profiles from the database until we find a compatible one
# This ensures that the sampled hardware can be realistically simulated on the physical machine



def _generate_gpu_sample(localinfo) -> list[str]:
    gpu_df = pd.read_csv("./bouquetfl/hardwareconf/gpus.csv")
    probabilities = gpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    tries = 0
    while not sample_compatible:
        tries += 1
        sampled_gpu = np.random.choice(
            gpu_df["gpu name"], p=probabilities, replace=True
        )
        gpu_info = gpu_df[gpu_df["gpu name"] == sampled_gpu].iloc[0]
        if (
            (gpu_info["CUDA cores"] <= localinfo.get("gpu_cores"))
            and (gpu_info["Clock speed"] <= localinfo.get("gpu_clock_speed"))
            and (gpu_info["Memory (GB)"] <= localinfo.get("gpu_memory"))
            and (gpu_info["Memory Speed"] <= localinfo.get("gpu_memory_speed"))
        ):
            sample_compatible = True
        if tries > 50:
            print("Could not find compatible GPU after 50 tries, using fallback GPU.")
            sampled_gpu = "GeForce GTX 1050"  # Fallback GPU
            sample_compatible = True
    return sampled_gpu


def _generate_cpu_sample(localinfo) -> list[str]:
    cpu_df = pd.read_csv("./bouquetfl/hardwareconf/cpus.csv")
    probabilities = cpu_df["shss"].astype(float)
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    tries = 0
    while not sample_compatible:
        tries += 1
        sampled_cpu = np.random.choice(
            cpu_df["cpu name"], p=probabilities, replace=True
        )
        cpu_info = cpu_df[cpu_df["cpu name"] == sampled_cpu].iloc[0]
        if len(cpu_info["cores"].split(" ")) > 1:
            num_cores = cpu_info["cores"].split(" ")[0]
        else:
            num_cores = cpu_info["cores"]
        if len(cpu_info["core clock"].split(" ")) > 1:
            clock_speed = 1000 * float(cpu_info["core clock"].split(" ")[0])
        else:
            clock_speed = 1000 * float(cpu_info["core clock"])
        if int(num_cores) <= localinfo.get("cpu_cores") and float(clock_speed) <= localinfo.get("cpu_clock_speed"):
            sample_compatible = True
        if tries > 50:
            print("Could not find compatible CPU after 50 tries, using fallback CPU.")
            sampled_cpu = "Ryzen 3 1200"  # Fallback CPU
            sample_compatible = True
    return sampled_cpu


def _generate_ram_sample(localinfo) -> int:
    ram_options = [4, 8, 12, 16, 24, 32, 48, 64]
    probabilities = np.array(
        [0.0162, 0.0838, 0.0261, 0.4149, 0.0185, 0.3593, 0.0109, 0.0435]
    )  # Directly sampled from Steam hardware survey
    probabilities = probabilities / np.sum(probabilities)
    sample_compatible = False
    ram = localinfo.get("ram_gb", None)
    if ram <= 4:
        return ram.tolist()
    while not sample_compatible:
        sampled_ram = np.random.choice(ram_options, p=probabilities, replace=True)
        if sampled_ram <= ram:
            sample_compatible = True
    return sampled_ram.tolist()


def generate_hardware_config(num_clients: int) -> None:
    client_hardware = {}

    local_info = profile_local_hardware()
    print(local_info)

    for client_id in range(num_clients):
        gpu, cpu, ram = (
            _generate_gpu_sample(local_info),
            _generate_cpu_sample(local_info),
            _generate_ram_sample(local_info),
        )
        client_hardware[f"client_{client_id}"] = {
            "gpu": gpu,
            "cpu": cpu,
            "ram_gb": ram,
        }
    with open("./bouquetfl/config/federation_client_hardware.yaml", "w") as hardware_file:
        yaml.dump(client_hardware, hardware_file)
