"""
emulation_engine.py — standalone hardware emulation engine.

Applies ALL hardware constraints for a target client profile, then spawns
training_worker.py as a restricted subprocess and waits for it to finish.

No Flower dependency — can be called from client_app.py or directly from a
script for standalone profiling.

Hardware restrictions applied here
-----------------------------------
  GPU clocks      : pynvml (no subprocess) — system-wide, inherited by child
  GPU memory      : torch.cuda.set_per_process_memory_fraction inside worker
  GPU thread %    : CUDA_MPS_ACTIVE_THREAD_PERCENTAGE env var passed to child
  CPU frequency   : power_clock_tools.set_cpu_limit (sysfs via sudo)
  CPU core count  : os.sched_setaffinity inside worker (set via config)
  RAM             : systemd-run --scope -p MemoryMax= (cgroup, reliable)
"""

import json
import os
import subprocess

from bouquetfl.core import power_clock_tools as pct


# ---------------------------------------------------------------------------
# MPS helpers
# ---------------------------------------------------------------------------

MPS_LOG_DIR = "/tmp/nvidia-mps"

def _start_mps() -> None:
    # Stop any leftover MPS server from a previous crashed run before starting fresh
    _stop_mps()
    os.makedirs(MPS_LOG_DIR, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG_DIR
    subprocess.Popen(["nvidia-cuda-mps-control", "-d"], env=env, stderr=subprocess.DEVNULL).wait()
    print("[engine] MPS server started")


def _stop_mps() -> None:
    env = os.environ.copy()
    env["CUDA_MPS_LOG_DIRECTORY"] = MPS_LOG_DIR
    subprocess.run(
        ["nvidia-cuda-mps-control"],
        input="quit\n",
        text=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("[engine] MPS server stopped")


# ---------------------------------------------------------------------------
# GPU clock helpers — nvidia-smi via pct.run() (sudo -S, uses stored password)
# ---------------------------------------------------------------------------

def _lock_clocks(gpu_index: int, clock_mhz: int, mem_clock_mhz: int) -> None:
    pct.lock_gpu_clocks(gpu_index, clock_mhz, clock_mhz)
    pct.lock_gpu_memory_clocks(gpu_index, mem_clock_mhz, mem_clock_mhz)


def _reset_clocks(gpu_index: int) -> None:
    pct.reset_gpu_clocks(gpu_index)
    pct.reset_gpu_memory_clocks(gpu_index)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_emulation(
    config: dict,
    hardware_profile: dict,
    local_hw: dict,
    input_params_path: str | None = None,
    output_params: bool = True,
) -> dict | None:
    """
    Run one training round inside a fully hardware-restricted subprocess.

    Parameters
    ----------
    config : dict
        All settings for the worker. Required keys:
            "task"         — path to the task module file (e.g. "task/cifar10.py")
            "client_id"    — partition ID for dataset loading
            "num-clients"  — total number of clients
            "batch-size"   — training batch size
            "local-epochs" — local training epochs
            "learning-rate"
        Optional keys:
            "server-round", "num-server-rounds"
        Hardware keys are injected automatically by this function.
    hardware_profile : dict
        Dict with keys "gpu" (str), "cpu" (str), "ram_gb" (int).
        Hardware specs are looked up from the CSV databases using these names.
    local_hw : dict
        Local machine hardware info from localinfo.get_all_local_info().
        Required keys: "gpu_cores", "cpu_cores", "cpu_clock_speed".
    input_params_path : str | None
        Path to a .pt file containing the initial model state_dict.
        If None, the task's get_initial_state_dict() is used inside the worker.
    output_params : bool
        If True, load and return the updated state_dict after training.

    Returns
    -------
    tuple[dict, dict | None]
        (timing, state_dict) where timing contains "data_load_time",
        "train_time", and "oom". state_dict is None if output_params=False
        or training failed.
    """

    client_id       = config["client_id"]
    task_module_path = config["task"]

    # ------------------------------------------------------------------
    # 1. Unpack hardware profile
    # ------------------------------------------------------------------
    gpu    = hardware_profile["gpu"]
    cpu    = hardware_profile["cpu"]
    ram_gb = hardware_profile["ram_gb"]

    gpu_info  = pct.get_gpu_info(gpu)
    cpu_info  = pct.get_cpu_info(cpu)
    num_cores = cpu_info["cores"]

    print(
        f"\n[engine] client {client_id} — "
        f"GPU: {gpu}  CPU: {cpu}  RAM: {ram_gb} GB  cores: {num_cores}"
    )

    # ------------------------------------------------------------------
    # 2. Lock GPU clocks (system-wide — child process inherits the effect)
    # ------------------------------------------------------------------
    clock_mhz     = int(gpu_info["clock speed"])
    mem_clock_mhz = int(gpu_info["memory speed"])
    print(f"[engine] locking GPU clocks → core {clock_mhz} MHz / mem {mem_clock_mhz} MHz")
    _lock_clocks(0, clock_mhz, mem_clock_mhz)

    # ------------------------------------------------------------------
    # 3. CPU frequency limit (system-wide via cpupower / sysfs)
    #    num_cores is enforced inside the worker via os.sched_setaffinity
    # ------------------------------------------------------------------
    print(f"[engine] setting CPU max freq → {cpu_info['base clock']} MHz")
    pct.set_cpu_limit(cpu, local_hw)

    # ------------------------------------------------------------------
    # 4. Start MPS server
    # ------------------------------------------------------------------
    _start_mps()

    # ------------------------------------------------------------------
    # 5. CUDA MPS thread percentage
    # ------------------------------------------------------------------
    local_cores  = int(local_hw["gpu_cores"])
    target_cores = min(gpu_info["cuda cores"], local_cores)
    mps_pct      = 100.0 * target_cores / local_cores
    print(
        f"[engine] CUDA MPS thread % → {round(mps_pct, 2)}%"
        f"  ({target_cores}/{local_cores} CUDA cores)"
    )

    # ------------------------------------------------------------------
    # 6. Build config dict for the worker
    #    Sent via stdin — no tempfile needed.
    # ------------------------------------------------------------------
    worker_cfg = {
        **config,
        "gpu":           gpu,
        "cpu":           cpu,
        "ram_gb":        ram_gb,
        "gpu_memory_gb":   gpu_info["memory"],
        "num_cores":       num_cores,
        "input_params":    input_params_path,
        "output_params":   f"/tmp/params_updated_{client_id}.tp" if output_params else None,
    }

    out_params_path = worker_cfg["output_params"]

    # ------------------------------------------------------------------
    # 8. Build subprocess environment
    #    OMP/MKL thread counts are set here so the worker's PyTorch thread
    #    pools do not exceed the number of cores pinned by sched_setaffinity.
    # ------------------------------------------------------------------
    env = os.environ.copy()
    env["CUDA_MPS_LOG_DIRECTORY"]            = MPS_LOG_DIR
    env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(mps_pct)
    env["OMP_NUM_THREADS"]                   = str(num_cores)
    env["MKL_NUM_THREADS"]                   = str(num_cores)
    env["HF_DATASETS_NUM_THREADS"]           = "1"
    env["TOKENIZERS_PARALLELISM"]            = "false"

    # ------------------------------------------------------------------
    # 9. Build the subprocess command
    #    systemd-run enforces the RAM cgroup limit (most reliable approach)
    # ------------------------------------------------------------------
    cmd = [
        "systemd-run",
        "--user",
        "--scope",
        "-p", f"MemoryMax={ram_gb}G",
        "poetry", "run", "python",
        "bouquetfl/core/training_worker.py",
    ]

    # ------------------------------------------------------------------
    # 10. Spawn, pipe config via stdin, collect timing from stdout.
    #     stderr is inherited (None) so [worker] status lines appear live
    #     in the terminal while stdout is captured for the JSON result.
    # ------------------------------------------------------------------
    print("[engine] spawning training worker …")
    stdout_data = ""
    try:
        child = subprocess.Popen(
            cmd, env=env, text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        stdout_data, _ = child.communicate(input=json.dumps(worker_cfg))
        print(f"[engine] worker finished (exit code {child.returncode})")
    finally:
        # ------------------------------------------------------------------
        # 11. Reset hardware and stop MPS — always runs even if worker crashed
        # ------------------------------------------------------------------
        pct.reset_cpu_limit()
        _reset_clocks(0)
        _stop_mps()

    # ------------------------------------------------------------------
    # 12. Parse and print timing reported by the worker
    # ------------------------------------------------------------------
    timing = None
    try:
        timing = json.loads(stdout_data.strip().splitlines()[-1])
        print(
            f"[engine] data load: {timing['data_load_time']:.2f}s  "
            f"train: {timing['train_time']:.2f}s  "
            f"({'OOM' if timing.get('oom') else 'OK'})"
        )
    except (json.JSONDecodeError, KeyError, IndexError):
        print(f"[engine] WARNING: could not parse worker timing output: {stdout_data!r}")

    # ------------------------------------------------------------------
    # 13. Load and return output parameters
    # ------------------------------------------------------------------
    if not output_params or out_params_path is None:
        return timing, None

    if os.path.exists(out_params_path):
        import torch
        state_dict = torch.load(out_params_path, weights_only=True)
        os.remove(out_params_path)
        return timing, state_dict

    print(
        f"[engine] WARNING: no output params at {out_params_path} — "
        "training likely failed (OOM or worker error)."
    )
    return timing, None
