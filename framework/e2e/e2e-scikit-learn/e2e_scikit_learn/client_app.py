import warnings

import numpy as np
from e2e_scikit_learn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from flwr.app import Context
from flwr.client import NumPyClient, start_client
from flwr.clientapp import ClientApp

# Load MNIST dataset from https://www.openml.org/d/554
num_partitions = 10
X_train, X_test, y_train, y_test = utils.load_data(
    num_partitions=num_partitions, partition_id=np.random.choice(num_partitions)
)

# Create LogisticRegression Model
model = LogisticRegression(
    penalty="l2",
    max_iter=1,  # local epoch
    warm_start=True,  # prevent refreshing weights when fitting
)

# Setting initial parameters, akin to model.compile for keras models
utils.set_initial_params(model)


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):  # type: ignore
        return utils.get_model_parameters(model)

    def fit(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        # Ignore convergence failure due to low local epochs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)
        return utils.get_model_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):  # type: ignore
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, len(X_test), {"accuracy": accuracy}


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    # Start Flower client
    start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())
