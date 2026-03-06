# BouquetFL: Heterogeneous Hardware Emulation for Federated Learning

BouquetFL is a framework for simulating heterogeneous client hardware in Federated Learning using the Flower framework.
It allows researchers to emulate clients with different CPU, GPU, and RAM capabilities on a single physical machine by enforcing hardware-level resource constraints at runtime.

BouquetFL is designed for studying realistic cross-device federated learning scenarios where client devices differ widely in computational power without requiring access to large, heterogeneous hardware testbeds. Each simulated client runs sequentially under a configurable hardware profile, enabling controlled and reproducible experimentation.

## How it works

Before launching the federation, if no client profiles are set, every client is assigned a hardware profile (GPU, CPU, RAM) sampled from the [Steam Hardware Survey](https://store.steampowered.com/hwsurvey/). The emulation engine then enforces these constraints at the OS/driver level:

| Resource | Mechanism |
|----------|-----------|
| GPU clock speed | `nvidia-smi --lock-gpu-clocks` / `--lock-memory-clocks` |
| GPU compute (CUDA cores) | CUDA MPS active thread percentage (`nvidia-cuda-mps-control`) |
| GPU memory | `torch.cuda.set_per_process_memory_fraction` |
| CPU frequency | `cpupower frequency-set` |
| CPU cores | `os.sched_setaffinity` |
| RAM | `systemd-run --scope -p MemoryMax` (cgroups) |

## System requirements

Tested on:

- **OS**: Ubuntu 22.04 (systemd-based Linux required)
- **GPU**: NVIDIA GeForce RTX 4070 Super (any NVIDIA GPU with CUDA and MPS support should work)
- **CPU**: AMD Ryzen 7 1800X (any CPU supported by `cpupower` should work)
- **Python**: 3.12

### Required system packages

- **NVIDIA drivers** + **nvidia-smi** (GPU clock locking)
- **CUDA toolkit** (`nvidia-cuda-mps-control` for MPS thread limiting)
- **cpupower** (`sudo apt install linux-tools-common linux-tools-$(uname -r)`)
- **systemd** (for `systemd-run` RAM cgroup limits — standard on most Linux distros)
- **sudo access** (required for `nvidia-smi` clock locking and `cpupower`)
- **poetry** (Python package manager)

On first run, you will be prompted for your sudo password (stored locally via `keyring` for subsequent runs).

## Running the example

```bash
flwr run
```

This runs a CIFAR-10 federated learning experiment with 5 clients, each assigned a different hardware profile. Configuration is in `pyproject.toml`:

```toml
[tool.flwr.app.config]
num-server-rounds = 3
local-epochs = 1
experiment = "cifar10"
model = "resnet18"
batch-size = 2048
learning-rate = 0.01
num-clients = 5
```

### Custom hardware profiles

To manually specify client hardware instead of sampling, create `config/federation_client_hardware.toml`:

```toml
[client_0]
gpu = "GeForce RTX 3080"
cpu = "Ryzen 5 3600"
ram_gb = 16

[client_1]
gpu = "GeForce GTX 1650"
cpu = "Core i5-10400"
ram_gb = 8
```

GPU and CPU names must match entries in `hardwareconf/gpus.toml` and `hardwareconf/cpus.toml`.

## Project structure

```
├── pyproject.toml                  # Flower app config + dependencies
├── scripts/
│   ├── server_app.py               # Flower ServerApp
│   └── client_app.py               # Flower ClientApp
├── bouquetfl/
│   ├── core/
│   │   ├── emulation_engine.py     # Hardware emulation orchestrator
│   │   ├── training_worker.py      # Training subprocess
│   │   └── power_clock_tools.py    # GPU/CPU clock control
│   └── utils/
│       ├── localinfo.py            # Local hardware profiling
│       └── sampler.py              # Hardware config sampling
├── task/
│   └── cifar10.py                  # CIFAR-10 task (model, data, train/test)
└── hardwareconf/
    ├── gpus.toml                   # GPU database (Steam Hardware Survey)
    └── cpus.toml                   # CPU database (Steam Hardware Survey)
```
