import tomllib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

# Steam Hardware Survey based sampling for GPUs and CPUs: hardware stats for Windows Computers (94.84% of total)
# Source: https://store.steampowered.com/hwsurvey/processormfg/   (October 2025)
# To generate samples, we first check the current hardware capabilities of the physical machine
# Then, we sample hardware profiles from the database until we find a compatible one
# This ensures that the sampled hardware can be realistically simulated on the physical machine


def _load_gpus() -> list[dict]:
    with open("hardwareconf/gpus.toml", "rb") as f:
        return tomllib.load(f)["gpus"]


def _load_cpus() -> list[dict]:
    with open("hardwareconf/cpus.toml", "rb") as f:
        return tomllib.load(f)["cpus"]


def _generate_gpu_sample(local_hw: dict) -> str:
    gpu_mem       = local_hw.get("gpu_memory", None)
    gpu_clock     = local_hw.get("gpu_clock_speed", None)
    gpu_mem_speed = local_hw.get("gpu_memory_speed", None)
    gpu_cores     = local_hw.get("gpu_cores", None)

    gpus = _load_gpus()
    names         = [g["name"] for g in gpus]
    probabilities = np.array([g["shss"] for g in gpus], dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    gpu_by_name   = {g["name"]: g for g in gpus}

    sample_compatible = False
    tries = 0
    while not sample_compatible:
        tries += 1
        sampled_name = np.random.choice(names, p=probabilities)
        gpu_info = gpu_by_name[sampled_name]
        if (
            gpu_info["cuda_cores"]  <= gpu_cores
            and gpu_info["clock_speed"]  <= gpu_clock
            and gpu_info["memory_gb"]    <= gpu_mem
            and gpu_info["memory_speed"] <= gpu_mem_speed
        ):
            sample_compatible = True
        if tries > 50:
            print("Could not find compatible GPU after 50 tries, using fallback GPU.")
            sampled_name = "GeForce GTX 1050"  # Fallback GPU
            sample_compatible = True
    return sampled_name


def _generate_cpu_sample(local_hw: dict) -> str:
    cpu_cores = local_hw.get("cpu_cores", None)
    cpu_clock = local_hw.get("cpu_clock_speed", None)

    cpus = _load_cpus()
    names         = [c["name"] for c in cpus]
    probabilities = np.array([c["shss"] for c in cpus], dtype=float)
    probabilities = probabilities / np.sum(probabilities)
    cpu_by_name   = {c["name"]: c for c in cpus}

    sample_compatible = False
    tries = 0
    while not sample_compatible:
        tries += 1
        sampled_name = np.random.choice(names, p=probabilities)
        cpu_info = cpu_by_name[sampled_name]
        # cores field: "2 / 4" (physical / logical) or plain "14"
        cores_str = cpu_info["cores"].split(" ")[0]
        # core_clock field: "2.4 to 3.3 GHz" or "3.6 GHz" — take the base value
        clock_str = cpu_info["core_clock"].split(" ")[0]
        clock_mhz = 1000 * float(clock_str)
        if int(cores_str) <= cpu_cores and clock_mhz <= cpu_clock:
            sample_compatible = True
        if tries > 50:
            print("Could not find compatible CPU after 50 tries, using fallback CPU.")
            sampled_name = "Ryzen 3 1200"  # Fallback CPU
            sample_compatible = True
    return sampled_name


def _generate_ram_sample(local_hw: dict) -> int:
    ram = local_hw.get("ram_gb", None)

    ram_options = [4, 8, 12, 16, 24, 32, 48, 64]
    probabilities = np.array(
        [0.0162, 0.0838, 0.0261, 0.4149, 0.0185, 0.3593, 0.0109, 0.0435]
    )  # Directly sampled from Steam hardware survey
    probabilities = probabilities / np.sum(probabilities)
    if ram <= 4:
        return int(ram)
    sample_compatible = False
    while not sample_compatible:
        sampled_ram = np.random.choice(ram_options, p=probabilities, replace=True)
        if sampled_ram <= ram:
            sample_compatible = True
    return sampled_ram.tolist()


def generate_hardware_config(num_clients: int, local_hw: dict) -> dict:
    print(
        f"Local hardware: GPU cores={local_hw.get('gpu_cores')}, "
        f"GPU clock={local_hw.get('gpu_clock_speed')} MHz, "
        f"GPU memory={local_hw.get('gpu_memory')} GB, "
        f"GPU memory speed={local_hw.get('gpu_memory_speed')} MHz, "
        f"CPU cores={local_hw.get('cpu_cores')}, "
        f"CPU clock={local_hw.get('cpu_clock_speed')} MHz, "
        f"RAM={local_hw.get('ram_gb')} GB"
    )
    client_hardware = {}
    for client_id in range(num_clients):
        gpu = _generate_gpu_sample(local_hw)
        cpu = _generate_cpu_sample(local_hw)
        ram = _generate_ram_sample(local_hw)
        client_hardware[f"client_{client_id}"] = {
            "gpu": gpu,
            "cpu": cpu,
            "ram_gb": ram,
        }
    return client_hardware
