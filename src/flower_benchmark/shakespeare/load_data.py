import json
import os
from typing import Tuple, Dict

import numpy as np


"""
data saved from leaf:https://github.com/TalwalkarLab/leaf/tree/master/data/shakespeare
using: ./preprocess.sh -s niid --sf 1.0 -k 0 -t sample -tf 0.8 (full-sized dataset)
for full datasets and 0.8 split for train and test data 
and saved in /flower_benchmark/dataset/shakespeare/ 
"""


def read_data(train_data_dir, test_data_dir):
    """parses data in given train and test data directories
    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        clients.extend(cdata["users"])
        if "hierarchies" in cdata:
            groups.extend(cdata["hierarchies"])
        train_data.update(cdata["user_data"])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
        test_data.update(cdata["user_data"])

    clients = list(sorted(train_data.keys()))

    return clients, groups, train_data, test_data


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return y_batch


def load_data(
    train_data_dir, test_data_dir, client_id: int
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    clients, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)
    client_name = clients[client_id]
    train_data = train_data[client_name]
    test_data = test_data[client_name]
    x_train = train_data["x"]
    y_train = train_data["y"]
    x_test = test_data["x"]
    y_test = test_data["y"]
    x_train = [word_to_indices(word) for word in x_train]
    x_train = np.array(x_train)
    x_test = [word_to_indices(word) for word in x_test]
    x_test = np.array(x_test)
    y_train = [letter_to_vec(c) for c in y_train]
    y_train = np.array(y_train)
    y_test = [letter_to_vec(c) for c in y_test]
    y_test = np.array(y_test)
    return (x_train, y_train), (x_test, y_test)


# --------------------------------------------------------------------------------
# utils for shakespeare dataset

ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)
NUM_LETTERS = len(ALL_LETTERS)
print(NUM_LETTERS)


def _one_hot(index, size):
    """returns one-hot vector with given size and value 1 at given index
    """
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    """returns one-hot representation of given letter
    """
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    """returns a list of character indices
    Args:
        word: string
    
    Return:
        indices: int list with length len(word)
    """
    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices
