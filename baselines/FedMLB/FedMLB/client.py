"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import flwr as fl
from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import tensorflow as tf


# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(
#         device=gpu, enable=True
#     )

# warnings.filterwarnings("ignore")


class TFClient(fl.client.NumPyClient):
    def __init__(
        self,
        # train_ds is a tf.data.Dataset
        train_ds: tf.data.Dataset,
        # model is a tf.keras.Model
        model: tf.keras.Model,
        num_examples_train: int,
        algorithm: str,
    ):
        self.model = model
        self.train_ds = train_ds
        self.num_examples_train = num_examples_train
        self.algorithm = algorithm

    def get_parameters(self, config: Dict[str, Scalar]):
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        epochs: int = config["local_epochs"]
        current_round: int = config["current_round"]
        exp_decay: int = config["exp_decay"]
        lr_client_initial: int = config["lr_client_initial"]
        if current_round > 1:
            lr_client = lr_client_initial * (exp_decay ** current_round)
            # During training, update the learning rate as needed
            tf.keras.backend.set_value(self.model.optimizer.lr, lr_client)

        # Update local model parameters
        if self.algorithm in ["FedMLB", "FedAvg+KD"]:
            self.model.local_model.set_weights(parameters)
            self.model.global_model.set_weights(parameters)
        elif self.algorithm in ["FedAvg"]:
            self.model.set_weights(parameters)

        # Get hyperparameters for this round
        # batch_size: int = config["batch_size"]
        # the dataset is already batched, so there is no need to retrieve the batch size

        # in model.fit it is not mandatory to specify
        # batch_size if the dataset is already batched
        # as in our case
        results = self.model.fit(self.train_ds, epochs=epochs)

        parameters_prime = self.model.get_weights()
        num_examples_train = self.num_examples_train
        # print(type(parameters_prime))
        # print(type(results.history))
        return parameters_prime, int(num_examples_train), results.history

# if __name__ == "__main__":
#     """Test that flower client is instantiated and correctly runs."""
#     client = client_fn(2)
#     model = fedmlb_models.create_resnet18(num_classes=100, input_shape=(None, 32, 32, 3), norm="group")
#     client.fit(model.get_weights(), {})
