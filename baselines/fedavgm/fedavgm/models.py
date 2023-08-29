"""CNN model architecture."""

from flwr.common import ndarrays_to_parameters
from keras.optimizers import SGD
from keras.regularizers import l2
from tensorflow import keras


def cnn(input_shape, num_classes):
    """CNN Model from (McMahan et. al., 2017).

    Communication-efficient learning of deep networks from decentralized data
    """
    input_shape = tuple(input_shape)

    weight_decay = 0.004
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(
                64,
                (5, 5),
                padding="same",
                activation="relu",
            ),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((3, 3), strides=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(
                384, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(
                192, activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = SGD(learning_rate=0.1)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model


def model_to_parameters(model):
    """Retrieve model weigths and convert to ndarrays."""
    return ndarrays_to_parameters(model.get_weights())
