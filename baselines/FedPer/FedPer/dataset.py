"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from FedPer.dataset_preparation import DATASETS, randomly_assign_classes

# working dir is two up
WORKING_DIR = Path(__file__).resolve().parent.parent
FL_BENCH_ROOT = WORKING_DIR.parent

sys.path.append(FL_BENCH_ROOT.as_posix())


def dataset_main(config: dict) -> None:
    dataset_name = config["name"].lower()
    dataset_folder = Path(WORKING_DIR, "datasets")
    dataset_root = Path(dataset_folder, dataset_name)

    if not os.path.isdir(dataset_root):
        os.makedirs(dataset_root)

    partition = {"separation": None, "data_indices": None}

    if dataset_name in ["cifar10", "cifar100"]:
        dataset = DATASETS[dataset_name](dataset_root, config)

        # randomly assign classes
        assert config["num_classes"] > 0, "Number of classes must be positive"
        config["num_classes"] = max(1, min(config["num_classes"], len(dataset.classes)))
        partition, stats = randomly_assign_classes(
            dataset=dataset,
            client_num=config["num_clients"],
            class_num=config["num_classes"],
        )
    else:
        raise RuntimeError("Please implement the dataset preparation for your dataset.")

    if partition["separation"] is None:
        clients_4_train = list(range(config["num_clients"]))
        clients_4_test = list(range(config["num_clients"]))

        partition["separation"] = {
            "train": clients_4_train,
            "test": clients_4_test,
            "total": config["num_clients"],
        }

    if config["name"] in ["cifar10", "cifar100"]:
        for client_id, idx in enumerate(partition["data_indices"]):
            if config["split"] == "sample":
                num_train_samples = int(len(idx) * config["fraction"])

                np.random.shuffle(idx)
                idx_train, idx_test = idx[:num_train_samples], idx[num_train_samples:]
                partition["data_indices"][client_id] = {
                    "train": idx_train,
                    "test": idx_test,
                }
            else:
                if client_id in clients_4_train:
                    partition["data_indices"][client_id] = {"train": idx, "test": []}
                else:
                    partition["data_indices"][client_id] = {"train": [], "test": idx}
    with open(dataset_root / "partition.pkl", "wb") as f:
        pickle.dump(partition, f)

    with open(dataset_root / "all_stats.json", "w") as f:
        json.dump(stats, f)

    # with open(dataset_root / "args.json", "w") as f:
    #    json.dump(prune_args(config), f)
