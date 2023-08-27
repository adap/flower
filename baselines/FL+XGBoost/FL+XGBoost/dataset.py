"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""
from dataset_preparation import _download_data,datafiles_fusion,train_test_split,modify_labels
from torch.utils.data import Dataset
from flwr.common import NDArray, NDArrays
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

def load_single_dataset(task_type,dataset_name,train_ratio=.75):
    datafiles_paths=_download_data(dataset_name)
    X,Y=datafiles_fusion(datafiles_paths)
    X_train,y_train,X_test,y_test=train_test_split(X,Y,train_ratio=train_ratio)
    if task_type.upper()=="BINARY":
        y_train,y_test=modify_labels(y_train,y_test)
    return X_train,y_train,X_test,y_test

