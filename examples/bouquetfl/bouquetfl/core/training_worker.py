"""
training_worker.py — training subprocess for bouquetfl emulation.

Spawned by emulation_engine.py inside a hardware-restricted environment.
Per-process hardware limits (CPU affinity, GPU memory fraction) are applied
here at startup because they cannot be set from the parent process.

Communication contract with emulation_engine
---------------------------------------------
  stdin  : JSON config dict (written by engine via communicate())
  stdout : single JSON line with timing results — read by engine
  stderr : human-readable status messages — flows live to terminal

No Flower dependency.
"""

import importlib.util
import json
import os
import sys
import timeit
import warnings

warnings.filterwarnings("ignore", message=".*fork.*",   category=DeprecationWarning)
warnings.filterwarnings("ignore", message="co_lnotab",  category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*", category=FutureWarning)

import pyarrow as pa
pa.set_cpu_count(1)

import torch


def _log(msg: str) -> None:
    """Print a status message to stderr so it appears live in the terminal
    without polluting the stdout channel used for result passing."""
    print(msg, file=sys.stderr)


def main() -> None:
    # ------------------------------------------------------------------
    # Read config from stdin (sent by emulation_engine via communicate())
    # ------------------------------------------------------------------
    cfg = json.load(sys.stdin)

    client_id        = cfg["client_id"]
    task_module_path = cfg["task"]
    num_cores        = cfg.get("num_cores", 1)
    input_params     = cfg.get("input_params")
    output_params    = cfg.get("output_params")

    # ------------------------------------------------------------------
    # Per-process hardware restrictions
    # These cannot be applied from emulation_engine (parent process);
    # they must run inside this subprocess.
    # ------------------------------------------------------------------

    # CPU affinity — pins this process and all its threads to num_cores
    # physical cores. This is the correct emulation of a limited CPU:
    # it constrains the training loop, PyTorch's inter-op thread pool,
    # OMP threads, and any other threads spawned by this process.
    os.sched_setaffinity(0, set(range(num_cores)))
    _log(f"[worker] CPU affinity → cores 0–{num_cores - 1}")

    # GPU memory fraction — torch limit is per-process, so it must be
    # set here rather than in the parent engine.
    if torch.cuda.is_available() and cfg.get("gpu_memory_gb"):
        total_memory = torch.cuda.get_device_properties(0).total_memory
        fraction = min(1.0, (cfg["gpu_memory_gb"] * 1024 ** 3) / total_memory)
        torch.cuda.set_per_process_memory_fraction(fraction, 0)
        _log(
            f"[worker] GPU memory → "
            f"{round(fraction * 100, 1)}% ({cfg['gpu_memory_gb']} GB target)"
        )

    # Disable TF32 for faithful emulation across all GPU generations
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # ------------------------------------------------------------------
    # Load task module dynamically from file path
    # ------------------------------------------------------------------
    spec   = importlib.util.spec_from_file_location("mltask", task_module_path)
    mltask = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mltask)
    _log(f"[worker] loaded task from {task_module_path}")

    # ------------------------------------------------------------------
    # Load initial model parameters
    # ------------------------------------------------------------------
    model = mltask.get_model()
    if input_params and os.path.exists(input_params):
        state_dict = torch.load(input_params, weights_only=True)
        model.load_state_dict(state_dict)
        _log(f"[worker] loaded params from {input_params}")
    else:
        model.load_state_dict(mltask.get_initial_state_dict())
        _log("[worker] using task's get_initial_state_dict()")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    t0 = timeit.default_timer()
    trainloader = mltask.load_data(
        client_id,
        num_clients=cfg["num-clients"],
        num_workers=num_cores,
        batch_size=cfg["batch-size"],
    )
    data_load_time = timeit.default_timer() - t0
    num_examples   = len(trainloader.dataset)
    _log(f"[worker] data loaded in {data_load_time:.2f}s  ({num_examples} examples)")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    device = (
        "cuda"
        if torch.cuda.is_available() and cfg.get("gpu") != "None"
        else "cpu"
    )

    _log(f"[worker] training on {device} …")
    oom = False
    t0  = timeit.default_timer()
    try:
        mltask.train(
            model=model,
            trainloader=trainloader,
            epochs=cfg["local-epochs"],
            device=device,
            lr=cfg["learning-rate"],
        )
        train_time = timeit.default_timer() - t0
        _log(f"[worker] finished in {train_time:.2f}s")

        if output_params:
            torch.save(model.state_dict(), output_params)
            _log(f"[worker] saved params to {output_params}")

    except torch.OutOfMemoryError:
        train_time   = -1.0
        oom          = True
        num_examples = 0  # exclude OOM client from FedAvg weighted averaging
        _log("[worker] OUT OF MEMORY — training aborted")
        if output_params:
            try:
                os.remove(output_params)
            except FileNotFoundError:
                pass

    # ------------------------------------------------------------------
    # Emit timing result to stdout for the engine to collect
    # ------------------------------------------------------------------
    print(json.dumps({
        "data_load_time": data_load_time,
        "train_time":     train_time,
        "oom":            oom,
        "num_examples":   num_examples,
    }))


if __name__ == "__main__":
    main()
