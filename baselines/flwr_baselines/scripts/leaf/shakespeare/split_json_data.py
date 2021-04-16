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
from typing import List, Optional, Tuple


def check_between_zero_and_one(value: str):
    fvalue = float(value)
    if fvalue < 0 or fvalue > 1:
        raise argparse.ArgumentTypeError(
            f"""Invalid partition fraction {fvalue}. This must be between 0 and 1."""
        )
    return fvalue


def split_json_and_save(
    list_datasets: List[Tuple[str, float]],
    path_to_json: PathLike,
    save_root: PathLike,
    users_list: Optional[List[str]] = [],
):
    """Splits LEAF generated datasets and creates individual client partitions.

    Args:
        list_datasets (List[Tuple[str, float]]): list containting dataset tags
            and fraction of dataset split.
        path_to_json (PathLike): Path to LEAF JSON file containing dataset.
        save_root (PathLike): Root directory where to save the individual client
            partition files.
    """
    new_users = []
    with open(path_to_json) as f:
        js = json.load(f)
        if not users_list:
            users_list = js["users"]
            print("Using previous list of users.")
        for user_idx, u in enumerate(users_list):
            new_users.append(u)
            x = js["user_data"][u]["x"]
            y = js["user_data"][u]["y"]
            num_samples = len(x)
            start_idx = 0
            for split_id, (dataset, fraction) in enumerate(list_datasets):
                end_idx = start_idx + int(fraction * num_samples)
                data = {}
                data["idx"] = user_idx
                data["character"] = u
                if split_id == len(list_datasets) - 1:  # Make sure we use last indices
                    end_idx = num_samples
                data["x"] = x[start_idx:end_idx]
                data["y"] = y[start_idx:end_idx]
                start_idx = end_idx

                save_dir = save_root / str(user_idx)
                save_dir.mkdir(parents=True, exist_ok=True)

                with open(save_dir / f"{dataset}.pickle", "wb") as f:
                    pickle.dump(data, f)

    return new_users


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Splits a LEAF Shakespeare train dataset into train/validation for each client
        and saves the clients' train/val/test dataset in their respective folder."""
    )
    parser.add_argument(
        "--save_root",
        type=str,
        required=True,
        help="Root folder where partitions will be save as {save_root}/client_id/{train,val,test}.pickle",
    )
    parser.add_argument(
        "--leaf_train_json",
        type=str,
        required=True,
        help="Complete path to JSON file containing the generated trainset for LEAF Shakespeare.",
    )
    parser.add_argument(
        "--val_frac",
        type=check_between_zero_and_one,
        required=True,
        default=0.2,
        help="Fraction of original trainset that will be used for validation.",
    )
    parser.add_argument(
        "--leaf_test_json",
        type=str,
        required=True,
        help="Complete path to JSON file containing the generated *testset* for LEAF Shakespeare.",
    )

    args = parser.parse_args()

    save_root = Path(args.save_root)

    # Split train dataset into train and validation
    # then save files for each client
    original_train_dataset = args.leaf_train_json
    train_frac = 1.0 - args.val_frac
    list_datasets = [("train", train_frac), ("val", args.val_frac)]
    users_list = split_json_and_save(
        list_datasets=list_datasets,
        path_to_json=original_train_dataset,
        save_root=save_root,
    )

    # Split and save the test files
    original_test_dataset = args.leaf_test_json
    list_datasets = [("test", 1.0)]
    split_json_and_save(
        list_datasets=list_datasets,
        path_to_json=original_test_dataset,
        save_root=save_root,
        users_list=users_list,
    )
