import tensorflow as tf
from model.wrapper import ModelWrapper


class LogisticRegressionModel(ModelWrapper):
    def __init__(self):
        x_shape = (30,)
        y_shape = 1

        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=tuple(x_shape)),
                tf.keras.layers.Dense(y_shape, activation="sigmoid"),
            ]
        )

        model.compile(loss="binary_crossentropy", optimizer="adam")
        super().__init__(model)
