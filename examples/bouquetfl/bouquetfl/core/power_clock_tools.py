import subprocess
import tomllib

import keyring
from keyrings.alt.file import PlaintextKeyring

keyring.set_keyring(PlaintextKeyring())

_SERVICE  = "power_clock_tools_service"
_USERNAME = "local_user"


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def _ensure_password() -> None:
    """Prompt for the sudo password and store it in keyring if not already saved."""
    if keyring.get_password(_SERVICE, _USERNAME) is None:
        keyring.set_password(_SERVICE, _USERNAME, input("Enter sudo password: "))
        print("Password saved securely.")


def _get_password() -> str:
    return keyring.get_password(_SERVICE, _USERNAME)


_ensure_password()


# ---------------------------------------------------------------------------
# Sudo helper
# ---------------------------------------------------------------------------

def run(cmd: list[str]) -> None:
    subprocess.run(
        ["sudo", "-S"] + cmd,
        input=_get_password() + "\n",
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


# ---------------------------------------------------------------------------
# GPU — clocks
# ---------------------------------------------------------------------------

def lock_gpu_clocks(gpu_index: int, min_mhz: int, max_mhz: int) -> None:
    run(["nvidia-smi", "-i", str(gpu_index), f"--lock-gpu-clocks={min_mhz},{max_mhz}"])


def reset_gpu_clocks(gpu_index: int) -> None:
    run(["nvidia-smi", "-i", str(gpu_index), "--reset-gpu-clocks"])


def lock_gpu_memory_clocks(gpu_index: int, min_mhz: int, max_mhz: int) -> None:
    run(["nvidia-smi", "-i", str(gpu_index), f"--lock-memory-clocks={min_mhz},{max_mhz}"])


def reset_gpu_memory_clocks(gpu_index: int) -> None:
    run(["nvidia-smi", "-i", str(gpu_index), "--reset-memory-clocks"])


# ---------------------------------------------------------------------------
# GPU — hardware info
# ---------------------------------------------------------------------------

def get_gpu_info(gpu_name: str) -> dict:
    with open("hardwareconf/gpus.toml", "rb") as f:
        data = tomllib.load(f)
    for r in data["gpus"]:
        if r["name"] == gpu_name:
            return {
                "name":             r["name"],
                "memory":           float(r["memory_gb"]),
                "memory type":      r["memory_type"],
                "memory bandwidth": float(r["memory_bw"]),
                "clock speed":      float(r["clock_speed"]),
                "memory speed":     float(r["memory_speed"]),
                "cuda cores":       int(r["cuda_cores"]),
            }
    raise ValueError(f"GPU '{gpu_name}' not found in database.")


# ---------------------------------------------------------------------------
# CPU — hardware info and limit setter
# ---------------------------------------------------------------------------

def get_cpu_info(cpu_name: str) -> dict:
    with open("hardwareconf/cpus.toml", "rb") as f:
        data = tomllib.load(f)
    for r in data["cpus"]:
        if r["name"] == cpu_name:
            # cores column: "2 / 4" (physical / logical) or plain "14" → take first token
            num_cores = int(str(r["cores"]).split()[0])

            # clock column: "2.4 to 3.3 GHz" → extract all numeric tokens
            clock_values = [
                float(x) for x in str(r["core_clock"]).split()
                if x.replace(".", "", 1).isdigit()
            ]
            base_clock  = clock_values[0]  * 1000  # GHz → MHz
            turbo_clock = clock_values[-1] * 1000

            return {
                "name":        r["name"],
                "cores":       num_cores,
                "base clock":  base_clock,
                "turbo clock": turbo_clock,
            }
    raise ValueError(f"CPU '{cpu_name}' not found in database.")


def set_cpu_limit(cpu_name: str, local_hw: dict) -> int:
    cpu = get_cpu_info(cpu_name)

    if cpu["cores"] > int(local_hw["cpu_cores"]):
        raise ValueError(
            f"CPU {cpu_name} has more cores ({cpu['cores']}) "
            f"than the local CPU ({local_hw['cpu_cores']})."
        )
    if cpu["base clock"] > float(local_hw["cpu_clock_speed"]):
        print(
            f"CPU {cpu_name} base clock ({cpu['base clock']} MHz) exceeds "
            f"local CPU ({local_hw['cpu_clock_speed']} MHz) — "
            "capping to local max."
        )

    run(["cpupower", "frequency-set", "-u", f"{cpu['base clock']}MHz"])
    print(f"CPU max freq set to {cpu['base clock']} MHz")
    return cpu["cores"]


def reset_cpu_limit() -> None:
    run(["cpupower", "frequency-set", "-g", "performance"])
