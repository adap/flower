"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np

from fedper.dataset_preparation import (
    call_dataset,
    flickr_preprocess,
    randomly_assign_classes,
)

# working dir is two up
WORKING_DIR = Path(__file__).resolve().parent.parent
FL_BENCH_ROOT = WORKING_DIR.parent

sys.path.append(FL_BENCH_ROOT.as_posix())


def dataset_main(config: dict) -> None:
    """Prepare the dataset."""
    dataset_name = config["name"].lower()
    dataset_folder = Path(WORKING_DIR, "datasets")
    dataset_root = Path(dataset_folder, dataset_name)

    if not os.path.isdir(dataset_root):
        os.makedirs(dataset_root)

    if dataset_name == "cifar10":
        dataset = call_dataset(dataset_name=dataset_name, root=dataset_root)

        # randomly assign classes
        assert config["num_classes"] > 0, "Number of classes must be positive"
        config["num_classes"] = max(1, min(config["num_classes"], len(dataset.classes)))
        # partition, stats = randomly_assign_classes(
        partition = randomly_assign_classes(
            dataset=dataset,
            client_num=config["num_clients"],
            class_num=config["num_classes"],
        )

        clients_4_train = list(range(config["num_clients"]))
        clients_4_test = list(range(config["num_clients"]))

        partition["separation"] = {
            "train": clients_4_train,
            "test": clients_4_test,
            "total": config["num_clients"],
        }
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
        with open(dataset_root / "partition.pkl", "wb") as pickle_file:
            pickle.dump(partition, pickle_file)

        # with open(dataset_root / "all_stats.json", "w") as f:
        #    json.dump(stats, f)

    elif dataset_name.lower() == "flickr":
        flickr_preprocess(dataset_root, config)
    else:
        raise RuntimeError("Please implement the dataset preparation for your dataset.")
