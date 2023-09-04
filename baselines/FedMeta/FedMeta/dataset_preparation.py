"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
import json
import os
from typing import List, Optional, Tuple, Dict, DefaultDict
from collections import defaultdict
import numpy as np
import torch
import random


# def _read_dataset() -> Dict[List, Dict, List]:
def _read_dataset(
        path: str
) -> Tuple[List, DefaultDict]:
    """Read (if necessary) and returns the leaf dataset.

    Returns
    -------
    Tuple[user, data[x,y]]
        The dataset for training and the dataset for testing Femnist.
    """
    users = []
    data = defaultdict(lambda: None)

    files = [f for f in os.listdir(path) if f.endswith('.json')]

    for file_name in files:
        with open(f'{path}/{file_name}') as f:
            dataset = json.load(f)
        users.extend(dataset['users'])
        data.update(dataset['user_data'])

    users = list(sorted(data.keys()))
    return users, data


def support_query_split(
        data: DefaultDict,
        label: List,
        support_ratio: int,
):
    np.random.seed(42)
    random_index = np.random.permutation(len(label))
    slice_index = int(len(label) * support_ratio)
    train_index = random_index[:slice_index]
    test_index = random_index[slice_index:]
    return data[train_index].tolist(), data[test_index].tolist(), label[train_index].tolist(), label[
        test_index].tolist()


def split_train_validation_test_clients(
        clients: List,
        train_rate: Optional[float] = 0.8,
        val_rate: Optional[float] = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    np.random.seed(42)
    train_rate = int(train_rate * len(clients))
    val_rate = int(val_rate * len(clients))
    test_rate = len(clients) - train_rate - val_rate

    index = np.random.permutation(len(clients))
    trans_numpy = np.asarray(clients)
    train_clients = trans_numpy[index[:train_rate]].tolist()
    val_clients = trans_numpy[index[train_rate:train_rate + val_rate]].tolist()
    test_clients = trans_numpy[index[train_rate + val_rate:]].tolist()

    return train_clients, val_clients, test_clients


def _partition_data(
        dir_path: str,
        support_ratio: Optional[float] = None,
) -> Tuple[Dict, Dict]:

    train_path = f'{dir_path}/train'
    test_path = f'{dir_path}/test'

    train_users, train_data = _read_dataset(train_path)
    test_users, test_data = _read_dataset(test_path)

    if support_ratio is None:
        train_dataset = {'users': [], 'user_data': {}, 'num_samples': []}
        test_dataset = {'users': [], 'user_data': {}, 'num_samples': []}

        for user in train_users:
            train_dataset['users'].append(user)
            train_dataset['user_data'][user] = {'x': train_data[user]['x'], 'y': train_data[user]['y']}
            train_dataset['num_samples'].append(len(train_data[user]['y']))

            test_dataset['users'].append(user)
            test_dataset['user_data'][user] = {'x': test_data[user]['x'], 'y': test_data[user]['y']}
            test_dataset['num_samples'].append(len(test_data[user]['y']))

        return train_dataset, test_dataset

    else:
        support_dataset = {'users': [], 'user_data': {}, 'num_samples': []}
        query_dataset = {'users': [], 'user_data': {}, 'num_samples': []}

        for user in train_users:
            print(f'now preprocessing user : {user}')
            all_x = np.asarray(train_data[user]['x'] + test_data[user]['x'])
            all_y = np.asarray(train_data[user]['y'] + test_data[user]['y'])
            sup_x, qry_x, sup_y, qry_y = support_query_split(all_x, all_y, support_ratio)

            support_dataset['users'].append(user)
            support_dataset['user_data'][user] = {'x': sup_x, 'y': sup_y}
            support_dataset['num_samples'].append(len(sup_y))

            query_dataset['users'].append(user)
            query_dataset['user_data'][user] = {'x': qry_x, 'y': qry_y}
            query_dataset['num_samples'].append(len(qry_y))

        return support_dataset, query_dataset
