import platform
import subprocess

import psutil
from numba import cuda
import yaml


def get_local_gpu_cores():

    cc_cores_per_SM_dict = {
        (2, 0): 32,
        (2, 1): 48,
        (3, 0): 192,
        (3, 5): 192,
        (3, 7): 192,
        (5, 0): 128,
        (5, 2): 128,
        (6, 0): 64,
        (6, 1): 128,
        (7, 0): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 9): 128,
        (9, 0): 128,
        (10, 0): 128,
        (12, 0): 128,
    }
    device = cuda.get_current_device()
    my_sms = getattr(device, "MULTIPROCESSOR_COUNT")
    my_cc = device.compute_capability
    cores_per_sm = cc_cores_per_SM_dict.get(my_cc)
    total_cores = cores_per_sm * my_sms
    return total_cores


def _get_local_gpu_info():
    query = "name,memory.total,clocks.max.graphics,clocks.max.memory"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True,
    )
    name, mem_total, max_graphics, max_mem = result.stdout.strip().split(", ")
    total_cores = get_local_gpu_cores()
    return {
        "gpu_name": name,
        "gpu_memory": int(int(mem_total) / 1024),
        "gpu_clock_speed": int(max_graphics),
        "gpu_memory_speed": int(max_mem),
        "gpu_cores": total_cores,
    }


def _get_local_cpu_info():
    cpu_info = {}
    cpu_info["cpu_cores"] = psutil.cpu_count(logical=False)
    cpu_info["cpu_clock_speed"] = psutil.cpu_freq().max
    return cpu_info


def _get_local_ram_info() -> int:
    ram = psutil.virtual_memory()
    available_ram_gb = int(ram.available / (1024**3))
    return {"ram_gb": available_ram_gb}


def _get_local_os_info():
    return {
        "os": platform.freedesktop_os_release()["NAME"]
        + " "
        + platform.freedesktop_os_release()["VERSION_ID"]
    }


def profile_local_hardware():
    local_info = {
        **_get_local_gpu_info(),
        **_get_local_cpu_info(),
        **_get_local_ram_info(),
        **_get_local_os_info(),
    }
    return local_info
