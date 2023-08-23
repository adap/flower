import flwr as fl
import math
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_val, y_val) -> None:
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=LOCAL_EPOCHS, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
  partitions = partition(CONCENTRATION)
  full_x_train_cid, full_y_train_cid = partitions[int(cid)]
  full_y_train_cid = to_categorical(full_y_train_cid, num_classes=num_classes)

  # Use 10% of the client's training data for validation
  split_idx = math.floor(len(full_x_train_cid) * 0.9)
  x_train_cid, y_train_cid = (
    full_x_train_cid[:split_idx],
    full_y_train_cid[:split_idx],
  )
  x_val_cid, y_val_cid = full_x_train_cid[split_idx:], full_y_train_cid[split_idx:]


  return FlowerClient(model, x_train_cid, y_train_cid, x_val_cid, y_val_cid)
