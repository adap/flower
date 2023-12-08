import sys

import numpy as np
import tensorflow as tf
from driver.driver import run_driver
from model.builder import Builder
from model.logistic_regression import LogisticRegressionModel
from model.wrapper import ModelWrapper


def multiply(input):
    return tf.keras.layers.Multiply()([input, np.array([2.0]).reshape([1, 1])])


def custom_preprocessing(input):
    # Mean and standard deviation normalization
    mean, variance = tf.nn.moments(input, axes=[1], keepdims=True)
    standardized_tensor = (input - mean) / tf.sqrt(variance)
    return standardized_tensor


def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(30,)),
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam")
    return model


def get_builder():
    wrapped_model = (
        ModelWrapper(get_model()) if "nn" in sys.argv else Cifar10CNNModel()
    )

    builder = Builder(wrapped_model)
    builder.add_function(multiply, "multiply01", [0, 1, 2])
    builder.add_function(custom_preprocessing, "custom01", [4, 5, 6])

    return builder


if __name__ == "__main__":
    address = "65.108.122.72"

    py_client = "py_client" in sys.argv
    new_model = "new" in sys.argv

    run_driver(get_builder(), new_model=new_model, py_client=py_client, address=address)
