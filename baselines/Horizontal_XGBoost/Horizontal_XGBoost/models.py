"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
from torch.utils.data import Dataset
from flwr.common import NDArray, NDArrays
from typing import Any, Dict, List, Optional, Tuple, Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset

def fit_XGBoost(
    config: DictConfig, task_type:str, X_train: NDArray,y_train: NDArray, n_estimators: int
) -> Union[XGBClassifier, XGBRegressor]:
    if task_type.upper() == "REG":
        tree = instantiate(config.XGBoost.regressor,n_estimators=n_estimators)
    elif task_type.upper() == "BINARY":
        tree = instantiate(config.XGBoost.classifier,n_estimators=n_estimators)
    tree.fit(X_train, y_train)
    return tree


def construct_tree_from_loader(
    dataset_loader: DataLoader, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset loader."""
    for dataset in dataset_loader:
        data, label = dataset[0], dataset[1]
    return fit_XGBoost(data, label, n_estimators, tree_type)