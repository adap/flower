"""Flower Server."""

import os

import joblib
from omegaconf import DictConfig
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .models import get_model


def get_on_fit_config_fn(conf: DictConfig):
    """Return fit_config_fn used in strategy."""

    def fit_config_fn(server_round: int):
        """Return the server's config file."""
        config = {
            "batch_size": conf.batch_size,
            "current_round": server_round,
            "local_epochs": conf.local_epochs,
            "total_rounds": conf.total_rounds,
        }

        return config

    return fit_config_fn


def get_evaluate_fn(
    testset: DataFrame, input_shape: int, num_classes: int, scaler_path: str
):
    """Return evaluate_fn used in strategy."""

    def evalaute_fn(server_round, parameters, config):
        """Evaluate the test set (if provided)."""
        if testset.empty:
            # this implies that testset is not used
            # and thus, included_testset from config file is False
            pass
        else:
            Y_test = testset[["type"]]
            enc_Y = LabelEncoder()
            Y_test = enc_Y.fit_transform(Y_test.to_numpy().reshape(-1))
            X_test = testset.drop(["type"], axis=1).to_numpy()

            # normalization
            # Check if the directory of the scaler exists and pick a scaler
            # of an arbitrary user. It's the same for all users.
            if not os.path.exists(scaler_path):
                scaler = StandardScaler()
                X_test = scaler.fit_transform(X_test)
            else:
                scaler = joblib.load(f"{scaler_path}/client_1/scaler.joblib")
                X_test = scaler.transform(X_test)

            model = get_model(input_shape, num_classes)
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(X_test, Y_test)
            return loss, {"accuracy": accuracy}

    return evalaute_fn
