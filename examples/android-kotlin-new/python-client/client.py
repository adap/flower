import sys

import flwr as fl
from task import SHAPE, LogisticRegressionModel, get_data


class Client(fl.client.NumPyClient):
    def __init__(self, model, data) -> None:
        self.model = model
        self.X_train, self.y_train, self.X_test, self.y_test = data

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.X_train, self.y_train, epochs=5)
        return (
            self.model.get_weights(),
            len(self.X_train),
            {},
        )

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        return loss, len(self.X_test), {"accuracy": accuracy}


if __name__ == "__main__":
    address = "65.108.122.72" if len(sys.argv) <= 1 else sys.argv[1]

    model = LogisticRegressionModel()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.build(SHAPE)

    data = get_data()

    fl.client.start_client(
        server_address=f"{address}:9092", client=Client(model, data).to_client()
    )
