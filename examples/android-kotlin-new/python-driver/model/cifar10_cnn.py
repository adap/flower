import tensorflow as tf
from model.wrapper import ModelWrapper


class Cifar10CNNModel(ModelWrapper):
    def __init__(self):
        x_shape = (32, 32, 3, )
        y_shape = 10

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=tuple(x_shape)),
                tf.keras.layers.Conv2D(6, 5, activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(16, 5, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=120, activation="relu"),
                tf.keras.layers.Dense(units=84, activation="relu"),
                tf.keras.layers.Dense(units=10, activation="softmax"),
            ]
        )

        model.compile(loss="categorical_crossentropy", optimizer="sgd")
        super().__init__(model)
