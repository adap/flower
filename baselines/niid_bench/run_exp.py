"""Script to run all experiments in parallel."""

import argparse
import subprocess
import time
from collections import deque
from itertools import product
from typing import List

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-processes", type=int, default=5)
parser_args = parser.parse_args()


def get_commands(dataset_name, partitioning, labels_per_client, seed):
    """Get commands for all experiments."""
    cmds = [
        (
            f"python -m niid_bench.main --config-name fedavg_base "
            f"partitioning={partitioning} "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
        (
            f"python -m niid_bench.main --config-name scaffold_base "
            f"partitioning={partitioning} "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
        (
            f"python -m niid_bench.main --config-name fedprox_base "
            f"partitioning={partitioning} "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
        (
            f"python -m niid_bench.main --config-name fedprox_base "
            f"partitioning={partitioning} "
            f"mu=0.1 "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
        (
            f"python -m niid_bench.main --config-name fedprox_base "
            f"partitioning={partitioning} "
            f"mu=0.001 "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
        (
            f"python -m niid_bench.main --config-name fedprox_base "
            f"partitioning={partitioning} "
            f"mu=1.0 "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
        (
            f"python -m niid_bench.main --config-name fednova_base "
            f"partitioning={partitioning} "
            f"dataset_seed={seed} "
            f"dataset_name={dataset_name} "
            f"labels_per_client={labels_per_client}"
        ),
    ]
    return cmds


dataset_names = ["cifar10", "mnist", "fmnist"]
partitionings = [
    "iid",
    "dirichlet",
    "label_quantity_1",
    "label_quantity_2",
    "label_quantity_3",
]

commands: deque = deque()
for partitioning, dataset_name in product(partitionings, dataset_names):
    labels_per_client = -1
    if "label_quantity" in partitioning:
        labels_per_client = int(partitioning.split("_")[-1])
        partitioning = "label_quantity"
    args = (dataset_name, partitioning, labels_per_client, parser_args.seed)
    cmds = get_commands(*args)
    for cmd in cmds:
        commands.append(cmd)

MAX_PROCESSES_AT_ONCE = parser_args.num_processes

# run max_processes_at_once processes at once with 10 second sleep interval
# in between those processes until all commands are done
processes: List = []
while len(commands) > 0:
    while len(processes) < MAX_PROCESSES_AT_ONCE and len(commands) > 0:
        cmd = commands.popleft()
        print(cmd)
        processes.append(subprocess.Popen(cmd, shell=True))
        # sleep for 10 seconds to give the process time to start
        time.sleep(10)
    for p in processes:
        if p.poll() is not None:
            processes.remove(p)
