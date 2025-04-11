"""statavg: A Flower Baseline."""

import tensorflow as tf


def get_model(input_shape: int, num_classes: int):
    """Return the model."""
    # creates a model with the given input shape and output: num_classes
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(128, activation="relu", input_dim=input_shape),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    opt = tf.keras.optimizers.Adam(
        learning_rate=0.002, beta_1=0.99, beta_2=0.999, epsilon=1e-08
    )
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]
    )
    return model
