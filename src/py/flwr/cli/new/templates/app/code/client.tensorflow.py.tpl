"""$project_name: A Flower / TensorFlow app."""

from flwr.client import NumPyClient, ClientApp

from $project_name.task import load_data, load_model


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(self.x_train, self.y_train, epochs=1, batch_size=32)
        return model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}


def client_fn(cid):
    # Load model and data
    net = load_model()
    x_train, y_train, x_test, y_test = load_data(cid)

    # Return Client instance
    return FlowerClient(net, x_train, y_train, x_test, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
