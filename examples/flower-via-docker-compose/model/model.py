import tensorflow as tf


# Class for the model. In this case, we are using the MobileNetV2 model from Keras
class Model:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        self.model = tf.keras.applications.MobileNetV2(
            (32, 32, 3), alpha=0.1, classes=10, weights=None
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def compile(self):
        self.model.compile(self.optimizer, self.loss_function, metrics=["accuracy"])

    def get_model(self):
        return self.model
