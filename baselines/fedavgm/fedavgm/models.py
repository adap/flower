"""CNN model architecture."""

from keras.optimizers import SGD
from keras.regularizers import l2
from tensorflow import keras


def create_model(input_shape, num_classes):
    """CNN Model from (McMahan et. al., 2017).

    Communication-efficient learning of deep networks from decentralized data
    """
    weight_decay = 0.004
    model = keras.Sequential(
        [
            keras.layers.Conv2D(
                32,
                (5, 5),
                activation="relu",
                input_shape=input_shape,
                kernel_regularizer=l2(weight_decay),
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Conv2D(
                64, (5, 5), activation="relu", kernel_regularizer=l2(weight_decay)
            ),
            keras.layers.MaxPooling2D(2, 2),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = SGD(decay=weight_decay)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    return model
