# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Partitioned versions of Spoken Keyword Detection dataset."""
# pylint: disable=invalid-name

import os
import pickle
import urllib.request
from typing import Tuple

from .dataset import (
    XY,
    PartitionedDataset,
    create_partitioned_dataset,
    log_distribution,
)


def download(filename: str, path: str) -> None:
    """Download hotkey dataset."""
    urls = {
        "hotkey_test_x.pkl": "https://www.dropbox.com/s/ve0g1m3wtuecb7r/hotkey_test_x.pkl?dl=1",
        "hotkey_test_y.pkl": "https://www.dropbox.com/s/hlihc8qchpo3hhj/hotkey_test_y.pkl?dl=1",
        "hotkey_train_x.pkl": "https://www.dropbox.com/s/05ym4jg8n7oi5qh/hotkey_train_x.pkl?dl=1",
        "hotkey_train_y.pkl": "https://www.dropbox.com/s/k69lhw5j02gsscq/hotkey_train_y.pkl?dl=1",
    }
    url = urls[filename]
    urllib.request.urlretrieve(url, path)
    print("Downloaded ", url)


def hotkey_load(dirname: str = "./data/hotkey/") -> Tuple[XY, XY]:
    """Load Hotkey dataset from disk."""
    files = [
        "hotkey_train_x.pkl",
        "hotkey_train_y.pkl",
        "hotkey_test_x.pkl",
        "hotkey_test_y.pkl",
    ]
    paths = []

    for f in files:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = os.path.join(dirname, f)
        if not os.path.exists(path):
            download(f, path)
        paths.append(path)

    with open(paths[0], "rb") as input_file:
        x_train = pickle.load(input_file)

    with open(paths[1], "rb") as input_file:
        y_train = pickle.load(input_file)

    with open(paths[2], "rb") as input_file:
        x_test = pickle.load(input_file)

    with open(paths[3], "rb") as input_file:
        y_test = pickle.load(input_file)

    return (
        (x_train[0:31000, :, :], y_train[0:31000]),
        (x_test[0:4000, :, :], y_test[0:4000]),
    )


def load_data(
    iid_fraction: float, num_partitions: int
) -> Tuple[PartitionedDataset, XY]:
    """Load partitioned version of FashionMNIST."""
    (xy_train_partitions, xy_test_partitions), xy_test = create_partitioned_dataset(
        hotkey_load(), iid_fraction, num_partitions
    )
    return (xy_train_partitions, xy_test_partitions), xy_test


if __name__ == "__main__":
    # Load a partitioned dataset and show distribution of examples
    for _num_partitions in [10, 50, 100]:
        for _fraction in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            (xy_train_par, xy_test_par), _ = load_data(_fraction, _num_partitions)
            print(f"\nfraction: {_fraction}; num_partitions: {_num_partitions}")
            log_distribution(xy_train_par)
            log_distribution(xy_test_par)
