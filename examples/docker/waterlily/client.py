import os
import sys

import flwr as fl
import tensorflow as tf


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# model.fit(x_train, y_train, epochs=1, batch_size=32)

# # Define Flower client
# class CifarClient(fl.client.NumPyClient):
#     def get_parameters(self, config):
#         return model.get_weights()

#     def fit(self, parameters, config):
#         model.set_weights(parameters)
#         model.fit(x_train, y_train, epochs=1, batch_size=32)
#         return model.get_weights(), len(x_train), {}

#     def evaluate(self, parameters, config):
#         model.set_weights(parameters)
#         loss, accuracy = model.evaluate(x_test, y_test)
#         return loss, len(x_test), {"accuracy": accuracy}


# # Start Flower client
# if __name__ == "__main__":
#     host = sys.argv[0]
#     fl.client.start_numpy_client(server_address=f"{host}:8080", client=CifarClient())
