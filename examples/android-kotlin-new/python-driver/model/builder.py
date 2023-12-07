from __future__ import annotations

import random
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
from flwr.common import Parameters
from model.wrapper import SAVED_MODEL_DIR


class AbstractBuilder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @property
    @abstractmethod
    def model(self) -> None:
        pass

    @property
    @abstractmethod
    def config(self) -> None:
        pass

    @property
    @abstractmethod
    def initial_parameters(self) -> None:
        pass

    @property
    @abstractmethod
    def tflite_model(self) -> None:
        pass

    @abstractmethod
    def add_multiply(self, name, index, multiplier) -> None:
        pass

    @abstractmethod
    def add_one_hot(self, name, index, num_tokens, output_mode) -> None:
        pass

    @abstractmethod
    def add_discretize(self, name, index, num_bins, epsilon) -> None:
        pass


class Builder(AbstractBuilder):
    def __init__(
        self, model_wrapper, tflite_file_path=f"{SAVED_MODEL_DIR}/model.tflite"
    ):
        self.model_wrapper = model_wrapper
        self._config = {}
        self.tflite_file = tflite_file_path

    def reset(self, model_wrapper):
        self.model_wrapper = model_wrapper
        self._config = {}

    @property
    def model(self):
        model = self.model_wrapper
        return model

    @property
    def tflite_model(self):
        model = self.model_wrapper.convert_to_tflite(self.tflite_file)
        return model

    @property
    def config(self):
        config = self._config
        config["model_id"] = str(hash(self.tflite_model))
        return config

    @property
    def initial_parameters(self):
        return self._ndarrays_to_parameters(self.model_wrapper.model.get_weights())

    def add_multiply(self, name, index, multiplier):
        def multiply(input):
            multiplier_array = np.array([multiplier])
            multiplier_array = tf.broadcast_to(
                multiplier_array, shape=tf.shape(input)[:1]
            )
            return tf.keras.layers.Multiply()([input, multiplier_array])

        self._add_to_builder(multiply, name, index)

    def add_one_hot(self, name, index, num_tokens, output_mode):
        def one_hot(input):
            layer = tf.keras.layers.CategoryEncoding(
                num_tokens=num_tokens, output_mode=output_mode
            )
            return layer(input)

        self._add_to_builder(one_hot, name, index)

    def add_discretize(self, name, index, num_bins, epsilon):
        def descretize(input):
            layer = tf.keras.layers.Discretization(num_bins=num_bins, epsilon=epsilon)
            layer.adapt(input)
            return layer(input)

        self._add_to_builder(descretize, name, index)

    def add_function(self, func, name, index):
        self._validate_func(func, index)
        self._add_to_builder(func, name, index)

    def _add_to_builder(self, func, name, index):
        func.__name__ = name
        if isinstance(index, list):
            value = ", ".join(str(i) for i in index)
        else:
            value = str(index)
        self._config.update({name: value})
        if "preprocess" not in self._config:
            self._config.update({"preprocess": name})
        else:
            self._config["preprocess"] += f", {name}"
        self.model_wrapper.add_preprocessing_func(func, index)

    def _validate_func(self, func, index):
        if not callable(func):
            raise ValueError("The provided argument is not a callable function.")

        N = random.randint(1, 100)
        input = tf.random.uniform((len(index), N), dtype=tf.float32)
        try:
            result = func(input)
        except Exception as e:
            raise ValueError(f"Function execution failed: {e}")

        if not isinstance(result, tf.Tensor):
            raise ValueError("The function does not return a Tensor.")

    def _ndarrays_to_parameters(self, ndarrays):
        tensors = [ndarray.tobytes() for ndarray in ndarrays]
        return Parameters(tensors=tensors, tensor_type="numpy.nda")
