"""Execute utility functions for fedht baseline."""

import numpy as np
from sklearn.linear_model import SGDClassifier
from flwr.common.typing import NDArrays

def set_model_params(model: SGDClassifier, params: NDArrays, cfg) -> SGDClassifier:
    """Sets the parameters of a sklean LogisticRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model

def get_model_parameters(model: SGDClassifier, cfg) -> NDArrays:
    """Returns the parameters of a sklearn LogisticRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params

def set_initial_params(model: SGDClassifier, cfg) -> None:
    """Sets initial parameters as zeros Required since model params are uninitialized
    until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer to
    sklearn.linear_model.LogisticRegression documentation for more information.
    """
    model.classes_ = np.arange(cfg.num_classes)

    model.coef_ = np.zeros((1, cfg.num_features))
    if model.fit_intercept:
        model.intercept_ = np.zeros((1,))


def create_log_reg_and_instantiate_parameters(cfg):
    """Helper function to create a LogisticRegression model."""
    model = SGDClassifier(
        loss='log_loss',
        learning_rate='constant',
        tol=.001,
        eta0=cfg.learning_rate,
        max_iter=cfg.num_local_epochs,  # local epoch
        warm_start=True,  # prevent refreshing weights when fitting,
    )

    # # Setting initial parameters, akin to model.compile for keras models
    # set_initial_params(model, cfg)
    return model
