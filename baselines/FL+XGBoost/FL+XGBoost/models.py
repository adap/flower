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

def construct_tree(
    dataset: Dataset, label: NDArray, n_estimators: int, tree_type: str
) -> Union[XGBClassifier, XGBRegressor]:
    """Construct a xgboost tree form tabular dataset."""
    if tree_type.upper() == "BINARY":
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

    elif tree_type.upper() == "REG":
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
