import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model


class SimpleMnistModel:
    def __init__(self, lr=0.001):
        self.lr = lr

    def run(self):
        model = self._build_model()
        return self._compile_model(model)

    def _build_model(self):
        input = Input(shape=(28, 28, 1))
        x = Conv2D(32, kernel_size=4, activation="relu")(input)
        x = MaxPooling2D()(x)
        x = Conv2D(16, kernel_size=4, activation="relu")(x)
        x = Flatten()(x)
        output = Dense(10, activation="softmax")(x)
        model = Model(inputs=input, outputs=output)
        return model

    def _compile_model(self, model):
        model.compile(
            optimizer=keras.optimizers.Adam(self.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model
