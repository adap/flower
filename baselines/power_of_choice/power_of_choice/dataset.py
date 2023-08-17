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
from typing import Optional, Tuple
import numpy as np
import tensorflow as tf
from omegaconf import DictConfig


def load_dataset(cid: str,) -> Tuple[np.ndarray, np.ndarray]:
    """Loads the dataset from file and returns the train set for a given client.

    Parameters
    ----------
    cid : str
        The client id.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing the train set for a given client.
    """

    folder = "fmnist"
    loaded_ds = tf.data.experimental.load(
            path=os.path.join(folder, cid), element_spec=None, compression=None, reader_func=None
        )
    
    # Unpack the loaded dataset into NumPy arrays
    x_train_cid = np.asarray(list(loaded_ds.map(lambda x, y: x)))
    y_train_cid = np.asarray(list(loaded_ds.map(lambda x, y: y)))
        
    return x_train_cid, y_train_cid
