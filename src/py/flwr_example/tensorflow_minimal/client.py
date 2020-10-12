import tensorflow as tf

import flwr as fl

if __name__ == "__main__":
    # Load and compile Keras model
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Define Flower client
    class CifarClient(fl.client.keras_client.KerasClient):
        def get_weights(self):  # type: ignore
            return model.get_weights()

        def fit(self, weights, config):  # type: ignore
            model.set_weights(weights)
            model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
            return model.get_weights(), len(x_train), len(x_train)

        def evaluate(self, weights, config):  # type: ignore
            model.set_weights(weights)
            loss, accuracy = model.evaluate(x_test, y_test)
            return len(x_test), loss, accuracy

    # Start Flower client
    fl.client.start_keras_client(server_address="[::]:8080", client=CifarClient())
