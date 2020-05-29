# Copyright 2020 The Flower Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
from logging import ERROR
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf

import flower as fl
from flower.logger import configure, log
from flower_benchmark.common import load_partition

# from flower_benchmark.dataset import tf_cifar_partitioned
from flower_benchmark.model import stacked_lstm
from flower_benchmark.tf_shakespeare.load_data import load_data
from flower_benchmark.tf_shakespeare.settings import SETTINGS, get_setting
from flower_benchmark.common import keras_evaluate
from . import DEFAULT_GRPC_SERVER_ADDRESS, DEFAULT_GRPC_SERVER_PORT, SEED

tf.get_logger().setLevel("ERROR")


class ShakespeareClient(fl.Client):
    def __init__(
        self,
        cid: str,
        model: tf.keras.Model,
        xy_train: Tuple[np.ndarray, np.ndarray],
        xy_test: Tuple[np.ndarray, np.ndarray],
        delay_factor: float,
        num_classes: int,
    ):

        super().__init__(cid)
        self.model = model

        self.ds_train = build_dataset(
            xy_train[0],
            xy_train[1],
            num_classes=num_classes,
            shuffle_buffer_size=len(xy_train[0]),
            augment=False,
        )
        self.ds_test = build_dataset(
            xy_test[0],
            xy_test[1],
            num_classes=num_classes,
            shuffle_buffer_size=0,
            augment=False,
        )

        self.num_examples_train = len(xy_train[0])
        self.num_examples_test = len(xy_test[0])
        self.delay_factor = delay_factor

    def get_parameters(self) -> fl.ParametersRes:
        # weights: fl.Weights = []
        # TODO get current weights from local model (not get from server?)
        parameters = fl.weights_to_parameters(self.model.get_weights())
        return fl.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.FitIns) -> fl.FitRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config: Dict[str, str] = ins[1]

        # Read training configuration
        epoch_global = int(config["epoch_global"])
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        timeout = int(config["timeout"])
        partial_updates = bool(int(config["partial_updates"]))

        # Use provided weights to update the local model
        self.model.set_weights(weights)

        # train model on local dataset
        completed, fit_duration, num_examples = custom_fit(
            model=self.model,
            dataset=self.ds_train,
            num_epochs=epochs,
            batch_size=batch_size,
            callbacks=[],
            delay_factor=self.delay_factor,
            timeout=timeout,
        )
        log(DEBUG, "client %s had fit_duration %s", self.cid, fit_duration)

        # Compute the maximum number of examples which could have been processed
        num_examples_ceil = self.num_examples_train * epochs

        # Return empty update if local update could not be completed in time
        if not completed and not partial_updates:
            parameters = fl.weights_to_parameters([])
            return parameters, num_examples, num_examples_ceil

        # Return the refined weights and the number of examples used for training
        parameters = fl.weights_to_parameters(self.model.get_weights())
        return parameters, num_examples, num_examples_ceil

        # TODO return tuple (trained parameters, training examples, training examples ceil)
        # return self.get_parameters(), num_examples, num_examples_ceil

    def evaluate(self, ins: fl.EvaluateIns) -> fl.EvaluateRes:
        weights: fl.Weights = fl.parameters_to_weights(ins[0])
        config: Dict[str, str] = ins[1]
        log(
            DEBUG,
            "evaluate on %s (examples: %s), config %s",
            self.cid,
            self.num_examples_test,
            config,
        )
        # TODO update local model with provided weights
        self.model.set_weights(weights)

        # TODO evaluate model on local dataset
        loss, _ = keras_evaluate(
            self.model, self.ds_test, batch_size=self.num_examples_test
        )

        # TODO return tuple (number of local evaluation examples, loss)
        return self.num_examples_test, loss


def parse_args() -> argparse.Namespace:
    """Parse and return commandline arguments."""
    parser = argparse.ArgumentParser(description="Flower Shakespeare")
    parser.add_argument(
        "--grpc_server_address",
        type=str,
        default=DEFAULT_GRPC_SERVER_ADDRESS,
        help="gRPC server address (IPv6, default: [::])",
    )
    parser.add_argument(
        "--log_host", type=str, help="HTTP log handler host (no default)",
    )
    parser.add_argument(
        "--setting", type=str, choices=SETTINGS.keys(), help="Setting to run.",
    )
    parser.add_argument(
        "--index", type=int, required=True, help="Client index in settings."
    )
    return parser.parse_args()


def main() -> None:
    """Load data, create and start client."""
    args = parse_args()

    client_setting = get_setting(args.setting).clients[args.index]

    # Configure logger
    configure(identifier=f"client:{client_setting.cid}", host=args.log_host)

    # Load model
    model = stacked_lstm(
        input_len=80, hidden_size=256, num_classes=80, embedding_size=80, seed=SEED
    )

    # dataset is already partitioned, since natural partition
    # TODO: how to determine the clients.cid, total 660 clients, not sure about client_setting.cid
    # need to download and preprocess the dataset, make sure to have 2 .json data one for training and one for testing
    # 660 clients

    xy_train, xy_test = load_data(
        "../dataset/shakespeare/train",
        "../dataset/shakespeare/test",
        client_setting.cid,
    )

    # Start client
    client = ShakespeareClient(
        client_setting.cid, model, xy_train, xy_test, client_setting.delay_factor, 80
    )

    fl.app.start_client(args.grpc_server_address, DEFAULT_GRPC_SERVER_PORT, client)


if __name__ == "__main__":
    # pylint: disable=broad-except
    try:
        main()
    except Exception as err:
        log(ERROR, "Fatal error in main")
        log(ERROR, err, exc_info=True, stack_info=True)

        # Raise the error again so the exit code is correct
        raise err
