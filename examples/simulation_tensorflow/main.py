import flwr as fl
from flwr.common.typing import Scalar
import ray
import tensorflow as tf
import pickle
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from utils import get_eval_fn, get_model, load_partition, partition_dataset

# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, partition_root: str):
        self.partition_dir = Path(partition_root) / str(cid)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

    def get_properties(self, ins):
        return self.properties

    def fit(self, parameters, config):
        # load partition dataset
        x_train, y_train = load_partition(self.partition_dir, self.cid, is_train=True)

        # send model to device
        model = get_model()
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32)

        # return local model and statistics
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        # load test  data for this client 
        with open(self.fed_dir / str(self.cid) / 'test.pickl', 'rb') as f:
            (x_test, y_test) = pickle.load(f)

        # send model to device
        model = self.load_model(parameters)

        # evaluate
        loss, accuracy = model.evaluate(x_test, y_test)

        return loss, len(x_test), {"accuracy": accuracy}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(5),
        "batch_size": str(64),
    }
    return config

# Start Ray simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 4. Starts a Ray-based simulation where a % of clients are sample each round.
# 5. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    total_num_clients = 100  # number of dataset partions (= number of total clients)
    client_resources = {"num_cpus": 1}  # each client will get allocated 1 CPUs

    # load CIFAR-10 dataset and create partitions
    trainset, testset = tf.keras.datasets.cifar10.load_data()
    fed_dir = Path('./federated/') # where to save partitions
    partition_dataset(trainset, fed_dir, num_partitions=total_num_clients, alpha=1000, prefix="train")

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1, # 0.1*100 = 10 clients per round
        min_fit_clients=10,
        min_available_clients=total_num_clients,  # All clients should be available
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),  # centralised testset evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return CifarRayClient(cid, fed_dir)

    # (optional) specify ray config
    ray_config = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=total_num_clients,
        client_resources=client_resources,
        num_rounds=5,
        strategy=strategy,
        ray_init_args=ray_config,
    )
