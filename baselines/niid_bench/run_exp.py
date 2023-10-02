import subprocess
import time
from collections import deque

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-processes", type=int, default=5)
parser_args = parser.parse_args()

def get_commands(dataset_name, partitioning, labels_per_client, seed):
    cmds = [
        f"python -m niid_bench.main --config-name fedavg_base partitioning={partitioning} dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}",
        f"python -m niid_bench.main --config-name scaffold_base partitioning={partitioning} dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}",
        f"python -m niid_bench.main --config-name fedprox_base partitioning={partitioning} dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}",
        f"python -m niid_bench.main --config-name fedprox_base partitioning={partitioning} mu=0.1 dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}",
        f"python -m niid_bench.main --config-name fedprox_base partitioning={partitioning} mu=0.001 dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}",
        f"python -m niid_bench.main --config-name fedprox_base partitioning={partitioning} mu=1.0 dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}",
        f"python -m niid_bench.main --config-name fednova_base partitioning={partitioning} dataset_seed={seed} dataset_name={dataset_name} labels_per_client={labels_per_client}"
    ]
    return cmds

dataset_names = ["cifar10", "mnist", "fmnist"]
partitionings = ["iid", "dirichlet", "label_quantity_1", "label_quantity_2", "label_quantity_3"]

from itertools import product

commands = deque()
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

# run max_processes_at_once processes at once with 1 second sleep interval in between those processes
# until all commands are done
processes = []
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