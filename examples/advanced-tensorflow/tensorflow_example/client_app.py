"""tensorflow-example: A Flower / Tensorflow app."""

import keras
from tensorflow_example.task import load_data, load_model

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ParametersRecord, RecordSet, array_from_numpy


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    """A simple client that showcases how to use the state.

    It implements a basic version of `personalization` by which
    the classification layer of the CNN is stored locally and used
    and updated during `fit()` and used during `evaluate()`.
    """

    def __init__(self, client_state: RecordSet, data, batch_size, local_epochs):
        self.client_state = client_state
        self.x_train, self.y_train, self.x_test, self.y_test = data
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.local_layer_name = "classification-head"

    def fit(self, parameters, config):
        """Train model locally.

        The client stores in its context the parameters of the last layer in the model
        (i.e. the classification head). The classifier is saved at the end of the
        training and used the next time this client participates.
        """

        # Instantiate model
        model = load_model(float(config["lr"]))

        # Apply weights from global models (the whole model is replaced)
        model.set_weights(parameters)

        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        self._load_layer_weights_from_state(model)

        model.fit(
            self.x_train,
            self.y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        # Save classification head to context's state to use in a future fit() call
        self._save_layer_weights_to_state(model)

        # Return locally-trained model and metrics
        return (
            model.get_weights(),
            len(self.x_train),
            {},
        )

    def _save_layer_weights_to_state(self, model):
        """Save last layer weights to state."""
        state_dict_arrays = {}
        # Get weights from the last layer
        layer_name = "dense"
        for variable in model.get_layer(layer_name).trainable_variables:
            state_dict_arrays[f"{layer_name}.{variable.name}"] = array_from_numpy(
                variable.numpy()
            )

        # Add to recordset (replace if already exists)
        self.client_state.parameters_records[self.local_layer_name] = ParametersRecord(
            state_dict_arrays
        )

    def _load_layer_weights_from_state(self, model):
        """Load last layer weights to state."""
        if self.local_layer_name not in self.client_state.parameters_records:
            return

        param_records = self.client_state.parameters_records
        list_weights = []
        for v in param_records[self.local_layer_name].values():
            list_weights.append(v.numpy())

        # Apply weights
        model.get_layer("dense").set_weights(list_weights)

    def evaluate(self, parameters, config):
        """Evaluate the global model on the local validation set.

        Note the classification head is replaced with the weights this client had the
        last time it trained the model.
        """
        # Instantiate model
        model = load_model()
        # Apply global model weights received
        model.set_weights(parameters)
        # Override weights in classification layer with those this client
        # had at the end of the last fit() round it participated in
        self._load_layer_weights_from_state(model)
        loss, accuracy = model.evaluate(self.x_test, self.y_test, verbose=0)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(context: Context):

    # Ensure a new session is started
    keras.backend.clear_session()
    # Load config and dataset of this ClientApp
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    data = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]

    # Return Client instance
    # We pass the state to persist information across
    # participation rounds. Note that each client always
    # receives the same Context instance (it's a 1:1 mapping)
    client_state = context.state
    return FlowerClient(client_state, data, batch_size, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
