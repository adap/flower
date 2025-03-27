"""statavg: A Flower Baseline."""

import json
import os
from typing import Dict, Tuple

import joblib
import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from omegaconf import OmegaConf
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from statavg.dataset import prepare_dataset
from statavg.model import get_model


# Define Flower client
# pylint: disable=too-many-instance-attributes
class FlowerClient(NumPyClient):
    """Client class that will implement StatAvg."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        trainset: DataFrame,
        save_path: str,
        val_ratio: float,
        client_id: int,
        learning_rate: float,
        strategy_name: str,
        local_epochs: int,
        batch_size: int,
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

        # path to save scalers
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.dir_path = os.path.join(script_dir, save_path, f"client_{self.client_id}")

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
        self.normalize_data()
        self.resample()

    def resample(self) -> None:
        """Perform resampling using SMOTE."""
        smo = SMOTE(random_state=42)
        self.x_train, self.y_train = smo.fit_resample(self.x_train, self.y_train)

    def normalize_data(self) -> None:
        """Scales/normalize data (z-score)."""
        # Check if the directory of the scaler exists
        if not os.path.exists(self.dir_path) or self._strategy_name == "fedavg":
            # this will be executed only on the first round iff 'statavg'
            # since the scaler directory does not exist yet
            scaler = StandardScaler()
            self.x_train = scaler.fit_transform(self.x_train)
            self.x_val = scaler.transform(self.x_val)
            self.scaler_first_r = scaler
        else:
            scaler = joblib.load(f"{self.dir_path}/scaler.joblib")
            self.x_train = scaler.transform(self.x_train)
            self.x_val = scaler.transform(self.x_val)
            # in case the directory already exists from previous experiments,
            # the following line is used to prevent error at the 1st round
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
        """Create a Dict with local statistics and sends it to the server.

        At the 1st round: Receives aggregated statistics from the server.
        At the 2nd round. Performs conventional local training.
        """
        # the client statistics are saved as a Dict[str, int] only at 1st round
        metrics = {}
        if config["current_round"] == 1 and self._strategy_name == "statavg":
            metrics = configure_metrics(self.scaler_first_r)

        # read the global statistics only at 2nd round
        if config["current_round"] == 2 and self._strategy_name == "statavg":
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

            # check if the directory exists, if not, create it
            if not os.path.exists(self.dir_path):
                os.makedirs(self.dir_path)

            #             Save the scaler/statistics
            # -------------------------------------------------
            # since Flower does not persist state within different rounds
            # the global statistics recieved by each client
            # are saved in a dedicated directory.
            # In subsequent rounds (>2), clients load the saved scalers and retrieve
            # the global statistics.
            # Note that scaling/normalization happens in __init()__
            # by invoking the method normalize_data().
            joblib.dump(scaler, f"{self.dir_path}/scaler.joblib")
            # ---------------------------------------------------

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

        return weights, len(self.x_train), metrics  # type: ignore

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate using the validation set: x_val."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_val, self.y_val, verbose=0)

        return loss, len(self.x_val), {"accuracy": accuracy}


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

    # Return Client instance
    return FlowerClient(
        trainset=trainset[partition_id],
        save_path=cfg.scaler_save_path,
        val_ratio=cfg.val_ratio,
        client_id=partition_id,
        learning_rate=cfg.learning_rate,
        strategy_name=cfg.strategy_name,
        local_epochs=cfg.local_epochs,
        batch_size=cfg.batch_size,
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
