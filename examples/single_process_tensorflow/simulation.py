import flwr as fl
import tensorflow as tf


def main() -> None:
    # Load CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Load model
    def load_model() -> tf.keras.Model:
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    # Define Flower client
    class CifarClient(fl.client.NumPyClient):
        def __init__(self, train_start, train_end, test_start, test_end):
            self.train_start = train_start
            self.train_end = train_end
            self.test_start = test_start
            self.test_end = test_end

        def get_parameters(self):
            return load_model().get_weights()

        def fit(self, parameters, config):
            model = load_model()
            model.set_weights(parameters)
            model.fit(
                x_train[self.train_start : self.train_end],
                y_train[self.train_start : self.train_end],
                epochs=1,
                batch_size=20,
            )
            return model.get_weights(), len(x_train[self.train_start : self.train_end])

        def evaluate(self, parameters, config):
            model = load_model()
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(
                x_test[self.test_start : self.test_end],
                y_test[self.test_start : self.test_end],
            )
            return len(x_test[self.test_start : self.test_end]), loss, accuracy

    # Start simulation
    num_clients = 500
    step_train = int(len(x_train) / num_clients)
    step_test = int(len(x_test) / num_clients)

    clients = []
    for i in range(num_clients):
        clients.append(
            CifarClient(
                train_start=i * step_train,
                train_end=(i + 1) * step_train,
                test_start=i * step_test,
                test_end=(i + 1) * step_test,
            )
        )
    print("Generated", len(clients), "clients")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.02,
        fraction_eval=0.01,
        min_fit_clients=5,
        min_eval_clients=3,
        min_available_clients=500,
    )
    fl.server.start_numpy_simulation(
        numpy_clients=clients, strategy=strategy, config={"num_rounds": 3}
    )


if __name__ == "__main__":
    main()
