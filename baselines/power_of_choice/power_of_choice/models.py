"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""
import tensorflow as tf

def create_MLP_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    return model

def create_CNN_model():
    # CNN on CIFAR10 for Image Classification: a deep convolutional neural network with 2
    # convolutional layers with max pooling and 4 hidden fully connected linear layers of units [120, 100, 84, 50].
    # The input is the flattened convolution output and the output is consisted of 10 units each of one of the 0-9
    # labels.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation="relu", input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation="relu"),
        tf.keras.layers.Dense(84, activation="relu"),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])

    return model