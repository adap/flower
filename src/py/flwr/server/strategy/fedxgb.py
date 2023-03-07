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
"""Federated XGBoost utility functions."""

from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch  # pylint: disable=E0401
import xgboost as xgb  # pylint: disable=E0401
from matplotlib import pyplot as plt  # pylint: disable=E0401
from torch.utils.data import DataLoader, Dataset  # pylint: disable=E0401
from xgboost import XGBClassifier, XGBRegressor  # pylint: disable=E0401

from flwr.common.typing import NDArray


def plot_xgbtree(tree: Union[XGBClassifier, XGBRegressor], n_tree: int) -> None:
    """Visualize the built xgboost tree."""
    xgb.plot_tree(tree, num_trees=n_tree)
    plt.rcParams["figure.figsize"] = [50, 10]
    plt.show()


def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset."""
    if tree_type == "BINARY":
        tree = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )

    elif tree_type == "REG":
        tree = xgb.XGBRegressor(
            objective="reg:squarederror",
            learning_rate=0.1,
            max_depth=8,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bylevel=1,
            colsample_bynode=1,
            colsample_bytree=1,
            alpha=5,
            gamma=5,
            num_parallel_tree=1,
            min_child_weight=1,
        )

    tree.fit(dataset, label)
    return tree


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return construct_tree(data, label, n_estimators, tree_type)


def single_tree_prediction(
    tree: Union[XGBClassifier, XGBRegressor], n_tree: int, dataset: NDArray
) -> Optional[NDArray]:
    """Extract the prediction result of a single tree in the xgboost tree
    ensemble."""
    # How to access a single tree
    # https://github.com/bmreiniger/datascience.stackexchange/blob/master/57905.ipynb
    num_t = len(tree.get_booster().get_dump())
    if n_tree > num_t:
        print(
            "The tree index to be extracted is larger than the total number of trees."
        )
        return None

    return tree.predict(  # type: ignore
        dataset, iteration_range=(n_tree, n_tree + 1), output_margin=True
    )


def tree_encoding(  # pylint: disable=R0914
    trainloader: DataLoader,
    client_trees: Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ],
    client_tree_num: int,
    client_num: int,
) -> Optional[Tuple[NDArray, NDArray]]:
    """Transform the tabular dataset into prediction results using the
    aggregated xgboost tree ensembles from all clients."""
    if trainloader is None:
        return None

    for local_dataset in trainloader:
        x_train, y_train = local_dataset[0], local_dataset[1]

    x_train_enc = np.zeros((x_train.shape[0], client_num * client_tree_num))
    x_train_enc = np.array(x_train_enc, copy=True)

    temp_trees: Any = None
    if isinstance(client_trees, list) is False:
        temp_trees = [client_trees[0]] * client_num
    elif isinstance(client_trees, list) and len(client_trees) != client_num:
        temp_trees = [client_trees[0][0]] * client_num
    else:
        cids = []
        temp_trees = []
        for i, _ in enumerate(client_trees):
            temp_trees.append(client_trees[i][0])  # type: ignore
            cids.append(client_trees[i][1])  # type: ignore
        sorted_index = np.argsort(np.asarray(cids))
        temp_trees = np.asarray(temp_trees)[sorted_index]

    for i, _ in enumerate(temp_trees):
        for j in range(client_tree_num):
            x_train_enc[:, i * client_tree_num + j] = single_tree_prediction(
                temp_trees[i], j, x_train
            )

    x_train_enc32: Any = np.float32(x_train_enc)
    y_train32: Any = np.float32(y_train)

    x_train_enc32, y_train32 = torch.from_numpy(
        np.expand_dims(x_train_enc32, axis=1)  # type: ignore
    ), torch.from_numpy(
        np.expand_dims(y_train32, axis=-1)  # type: ignore
    )
    return x_train_enc32, y_train32
