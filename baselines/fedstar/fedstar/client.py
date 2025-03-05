"""Required imports for client.py script."""

import gc
import logging
import os
from typing import Dict, List

import flwr
import tensorflow as tf

from fedstar.dataset import DataBuilder
from fedstar.models import Network, PslNetwork

os.environ["GRPC_VERBOSITY"] = "ERROR"

LOG_LEVEL = logging.ERROR


class AudioClient(
    flwr.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """AudioClient class act as a client for FedStar implementation.

    It extends NumpyClient feature because flower internally requires a NumpyClient.
    """

    # pylint: disable=too-many-arguments, too-many-locals, unexpected-keyword-arg
    def __init__(
        self,
        client_id,
        num_clients,
        dataset_dir,
        fedstar,
        parent_path,
        variance=0.25,
        batch_size=64,
        learning_rate=0.001,
        l_per=0.2,
        u_per=1.0,
        class_distribute: bool = False,
        balance_dataset: bool = False,
        mean_class_distribution=3,
        seed=2021,
        verbose=2,
    ):
        # Client Parameters
        self.client_id = client_id
        self.batch_size = batch_size
        self.verbose = verbose
        self.fedstar = fedstar
        self.aux_loss_weight = 0.5
        self.parent_path = parent_path
        # Load Clients Data
        (
            self.train_labelled,
            self.train_unlabelled,
            self.num_classes,
            self.num_batches,
        ) = DataBuilder.load_sharded_dataset(
            parent_path=self.parent_path,
            data_dir=dataset_dir,
            num_clients=num_clients,
            client=client_id,
            variance=variance,
            batch_size=batch_size,
            l_per=l_per,
            u_per=u_per,
            fedstar=fedstar,
            class_distribute=class_distribute,
            mean_class_distribution=mean_class_distribution,
            balance_dataset=balance_dataset,
            seed=seed,
        )
        self.num_examples_train = (
            self.num_batches * batch_size if self.train_labelled else 0
        )
        # Local Variables Initialize
        self.local_train_round = 0
        self.local_evaluate_round = 0
        self.weights = Network.get_init_weights(num_classes=self.num_classes)
        self.history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

        if self.fedstar:
            self.model = PslNetwork(
                num_classes=self.num_classes, aux_loss_weight=self.aux_loss_weight
            )
            self.model.compile(optimizer=tf.keras.optimizers.Adam(float(learning_rate)))
        else:
            self.model = Network(num_classes=self.num_classes).get_network()
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(float(learning_rate)),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True, name="loss"
                ),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            )  # type: ignore

        # tf.keras.backend.clear_session()

    def get_parameters(self, config):
        """Return the current client model weights."""
        return self.weights

    # pylint: disable=no-value-for-parameter
    def fit(self, parameters, config):
        """Update the current client model weights."""
        self.local_train_round += 1
        # Run Training Proccess
        if self.fedstar:
            self.model.set_weights(parameters)
            history = self.model.fit(
                (self.train_labelled, self.train_unlabelled),
                num_batches=self.num_batches,
                epochs=int(config["epochs"]),
                c_round=int(config["c_round"]),
                rounds=int(config["rounds"]),
                verbose=self.verbose,
            )
        else:
            self.model.set_weights(parameters)
            history = self.model.fit(
                self.train_labelled,
                batch_size=int(config["batch_size"]),
                epochs=int(config["epochs"]),
                verbose=self.verbose,
            )
        # Print Results
        print(
            f"""Client {self.client_id}
            finished {self.local_train_round}th round of training with
            loss {history.history['loss'][0]:.4f} and
            accuracy {history.history['accuracy'][0]:.4f}"""
        )
        # Clear Memory
        # tf.keras.backend.clear_session()
        gc.collect()
        return (
            self.model.get_weights(),
            self.num_examples_train,
            {"local_train_round": self.local_train_round},
        )

    def evaluate(self, parameters, config):
        """Essential function for Numpy Client but not required for FedStar."""
        raise NotImplementedError("Client Side evaluation not required.")


def set_logger_level():
    """Set the logging level for the 'flower' logger."""
    if "flower" in [
        repr(logging.getLogger(name))[8:].split(" ")[0]
        for name in logging.Logger.manager.loggerDict
    ]:
        logger = logging.getLogger("flower")
        logger.setLevel(LOG_LEVEL)
