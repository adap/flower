# Copyright 2021 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import json
import pickle
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def read_jsons_and_save(
    list_jsons: List[PathLike],
    save_root: PathLike,
    users_list: Optional[Dict[str, int]] = None,
):
    """Reads LEAF JSON files and stores user data as pickle files.

    Args:
        list_jsons (List[PathLike]): List of paths to LEAF FEMNIST dataset JSON files
        save_root (PathLike): Root directory where to save the files.
        users_list (Optional[Dict[str, int]], optional): Dictionary of users that have
            already been processed. Defaults to [].
    """
    if users_list is None:
        users_list = []

    counter: int = 0
    for path_to_json in list_jsons:
        with open(path_to_json, "r") as f:
            json_file = json.load(f)
        for user_idx, u in enumerate(json_file["users"]):
            global_id = user_idx + counter

            data = {}
            data["tag"] = u
            data["idx"] = str(global_id)
            data["x"] = json_file["user_data"][u]["x"]
            data["y"] = json_file["user_data"][u]["y"]

            save_dir = save_root / str(global_id)
            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_dir / f"{dataset}.pickle", "wb") as f:
                pickle.dump(data, f)

            counter += 1


def check_between_zero_and_one(value: str):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(
            f"""Invalid partition fraction {fvalue}. This must be between 0 and 1."""
        )
    return fvalue


def collect_all_users(list_jsons: List[PathLike]):
    all_users = []
    for path_to_json in paths_to_json:
        with open(path_to_json, "r") as f:
            json_file = json.load(f)
            all_users.append(sorted(json_file["users"]))

    return sorted((list(set(all_users))))


def split_json_and_save(
    list_datasets: List[Tuple[str, float]],
    paths_to_json: List[PathLike],
    save_root: PathLike,
):
    """Splits LEAF generated datasets and creates individual client partitions.

    Args:
        list_datasets (List[Tuple[str, float]]): list containting dataset tags
            and fraction of dataset split.
        paths_to_json (PathLike): Path to LEAF JSON files containing dataset.
        save_root (PathLike): Root directory where to save the individual client
            partition files.
    """
    user_count = 0
    for path_to_json in paths_to_json:
        with open(path_to_json, "r") as json_file:
            json_file = json.load(json_file)
            users_list = sorted(json_file["users"])
            num_users = len(users_list)
            for user_idx, user_str in enumerate(users_list):
                image = json_file["user_data"][user_str]["x"]
                label = json_file["user_data"][user_str]["y"]
                num_samples = len(label)
                start_idx = 0
                for split_id, (dataset, fraction) in enumerate(list_datasets):
                    end_idx = start_idx + int(fraction * num_samples)
                    data = {}
                    data["idx"] = user_idx + user_count
                    data["tag"] = user_str
                    if (
                        split_id == len(list_datasets) - 1
                    ):  # Make sure we use last indices
                        end_idx = num_samples
                    data["x"] = [
                        Image.fromarray(
                            np.uint8(
                                255
                                * np.asarray(img, dtype=np.float32).reshape((28, 28))
                            )
                        )
                        for img in image[start_idx:end_idx]
                    ]
                    data["y"] = label[start_idx:end_idx]
                    start_idx = end_idx

                    save_dir = save_root / str(user_idx + user_count)
                    save_dir.mkdir(parents=True, exist_ok=True)

                    with open(save_dir / f"{dataset}.pickle", "wb") as open_file:
                        pickle.dump(data, open_file)

        user_count += num_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Splits a LEAF FEMNIST train dataset into
                       train/validation for each client and saves the clients'
                       train/val/test dataset in their respective folder."""
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="""Root folder where partitions will be save as
                {save_root}/client_id/{train,val,test}.pickle""",
    )
    parser.add_argument(
        "--leaf_train_jsons_root",
        type=str,
        required=True,
        help="""Root directory to JSON files containing the
                generated trainset for LEAF FEMNIST.""",
    )
    parser.add_argument(
        "--val_frac",
        type=check_between_zero_and_one,
        required=True,
        default=0.2,
        help="Fraction of original trainset that will be used for validation.",
    )
    parser.add_argument(
        "--leaf_test_jsons_root",
        type=str,
        required=True,
        help="""Root folder to JSON file containing the generated *testset*
                for LEAF FEMNIST.""",
    )

    args = parser.parse_args()

    # Split train dataset into train and validation
    # then save files for each client
    original_train_datasets = sorted(
        list(Path(args.leaf_train_jsons_root).glob("**/*.json"))
    )
    train_frac = 1.0 - args.val_frac
    train_val_part_scheme = [("train", train_frac), ("val", args.val_frac)]
    split_json_and_save(
        list_datasets=train_val_part_scheme,
        paths_to_json=original_train_datasets,
        save_root=Path(args.save_root),
    )

    # Split and save the test files
    original_test_datasets = sorted(
        list(Path(args.leaf_test_jsons_root).glob("**/*.json"))
    )
    test_part_scheme = [("test", 1.0)]
    split_json_and_save(
        list_datasets=test_part_scheme,
        paths_to_json=original_test_datasets,
        save_root=Path(args.save_root),
    )
