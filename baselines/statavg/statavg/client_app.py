"""statavg: A Flower Baseline."""

import json
import pickle
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from omegaconf import OmegaConf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from flwr.client import ClientApp, NumPyClient
from flwr.common import ConfigRecord, Context
from flwr.common.typing import NDArrays, Scalar
from statavg.dataset import prepare_dataset
from statavg.model import get_model


# Define Flower client
# pylint: disable=too-many-instance-attributes
class FedAvgClient(NumPyClient):
    """Client class that will implement FedAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        trainset: DataFrame,
        val_ratio: float,
        client_id: int,
        learning_rate: float,
        strategy_name: str,
        local_epochs: int,
        batch_size: int,
        client_state,
    ) -> None:

        # load trainset
        self.data = trainset

        # client id
        self.client_id = client_id
        self.val_ratio = val_ratio
        self.learning_rate = learning_rate
        self._strategy_name = strategy_name
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_state = client_state

        # preprocess and split the dataset
        self.x_train, self.y_train, self.x_val, self.y_val = preprocess_and_split(
            self.data, val_ratio
        )

        # get the model
        self.model = get_model(self.x_train.shape[1], len(self.y_train.value_counts()))

        opt = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=0.99, beta_2=0.999, epsilon=1e-08
        )
        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
        )

        # encode labels
        self.encode_labels()

        # scale and resample
        self.scaler_first_r = None  # will be used only at the 1st round
        scaler, use_transform_only = self.determine_scaler()
        self.normalize_data(scaler, use_transform_only)
        self.resample()

    def determine_scaler(self):
        """Determine data scaler."""
        return StandardScaler(), False

    def resample(self) -> None:
        """Perform resampling using SMOTE."""
        smo = SMOTE(random_state=42)
        self.x_train, self.y_train = smo.fit_resample(self.x_train, self.y_train)

    def normalize_data(self, scaler, use_transform_only) -> None:
        """Normalise data."""
        if use_transform_only:
            self.x_train = scaler.transform(self.x_train)
        else:
            self.x_train = scaler.fit_transform(self.x_train)
        self.x_val = scaler.transform(self.x_val)
        self.scaler_first_r = scaler

    def encode_labels(self) -> None:
        """Encode the labels."""
        enc_y = LabelEncoder()
        self.y_train = enc_y.fit_transform(self.y_train.to_numpy().reshape(-1))
        self.y_val = enc_y.transform(self.y_val.to_numpy().reshape(-1))

    def get_parameters(self, config):
        """Get the training parameters."""
        return self.model.get_weights()

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Create a Dict with local statistics and sends it to the server."""
        # set weights and fit
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
        )

        weights = self.model.get_weights()  # type: ignore
        return weights, len(self.x_train), {}  # type: ignore

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate using the validation set: x_val."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)

        return loss, len(self.x_val), {"accuracy": accuracy}


class StatAvgClient(FedAvgClient):
    """Client class that will implement StatAvg."""

    def determine_scaler(self):
        """Determine scaler with client context."""
        if "scaler" not in self.client_state.config_records.keys():
            scaler = StandardScaler()
            use_transform_only = False
        else:
            scaler = pickle.loads(
                self.client_state.config_records["scaler"]["serialized_obj"]
            )
            use_transform_only = True
        return scaler, use_transform_only

    def fit(self, parameters, config):
        """Create a Dict with local statistics and sends it to the server.

        At the 1st round: Receives aggregated statistics from the server.
        At the 2nd round. Performs conventional local training.
        """
        # the client statistics are saved as a Dict[str, int] only at 1st round
        metrics = {}
        if config["current_round"] == 1:
            metrics = configure_metrics(self.scaler_first_r)

        # read the global statistics only at 2nd round
        if config["current_round"] == 2:
            for key, val in config.items():
                if val == 0:
                    stats_global = json.loads(key)
                else:
                    stats_global = None

            mean_global = np.array(stats_global["mean_global"])
            var_global = np.array(stats_global["var_global"])
            std_global = np.sqrt(var_global)

            scaler = StandardScaler()
            scaler.mean_ = mean_global
            scaler.var_ = var_global
            scaler.scale_ = std_global
            #             Save the scaler/statistics
            # -------------------------------------------------
            # In subsequent rounds (>2), clients load the saved scalers and retrieve
            # the global statistics.
            # Note that scaling/normalization happens in __init()__
            # by invoking the method normalize_data().
            serialized_scaler = pickle.dumps(scaler)
            self.client_state.config_records["scaler"] = ConfigRecord(
                {"serialized_obj": serialized_scaler}
            )
            # Sent to the server so that the
            # server can access the local scalers for eval.
            metrics = {"scaler": serialized_scaler}
            # ---------------------------------------------------

        weights, train_len, _ = super().fit(parameters, config)
        return weights, train_len, metrics


def preprocess_and_split(
    data: DataFrame, val_ratio: float
) -> Tuple[np.ndarray, DataFrame, np.ndarray, DataFrame]:
    """Preprocess and split data into train and val sets."""
    # keep label('type')
    Y = data[["type"]]

    # remove the label('type') from data
    data = data.drop(["type"], axis=1)

    # train-test splitting
    x_train, x_val, y_train, y_val = train_test_split(
        data, Y, test_size=val_ratio, stratify=Y, random_state=42
    )
    x_train = x_train.to_numpy()
    x_val = x_val.to_numpy()
    return x_train, y_train, x_val, y_val


def configure_metrics(scaler: StandardScaler) -> Dict[str, int]:
    """Transform the client's statistics to a Dict."""
    # insert statistical metrics into a dict
    stats = {"mean": scaler.mean_.tolist(), "var": scaler.var_.tolist()}

    # convert to json
    json_stats = json.dumps(stats)

    # 0 is a random int, will not be used anywhere.
    # Used just for consistency with Flower documentation.
    metrics = {json_stats: 0}

    return metrics


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    cfg = OmegaConf.create(context.run_config)
    partition_id = int(context.node_config["partition-id"])
    trainset, _ = prepare_dataset(
        cfg.num_clients, cfg.path_to_dataset, cfg.include_test, cfg.testset_ratio
    )
    client_state = context.state
    client_type = FedAvgClient if cfg.strategy_name == "fedavg" else StatAvgClient
    # Return Client instance
    return client_type(
        trainset=trainset[partition_id],
        val_ratio=cfg.val_ratio,
        client_id=partition_id,
        learning_rate=cfg.learning_rate,
        strategy_name=cfg.strategy_name,
        local_epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
        client_state=client_state,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
