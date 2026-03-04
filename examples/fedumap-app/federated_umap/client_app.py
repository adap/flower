"""
Federated UMAP — client-side Flower app.

Implements the client component from:
    "Federated t-SNE and UMAP for Distributed Data Visualization"
    Dong Qiao*, Xinxian Ma*, Jicong Fan†
    The 39th AAAI Conference on Artificial Intelligence (AAAI-25), 2025
    * equal contribution  † corresponding author
    https://ojs.aaai.org/index.php/AAAI/article/view/34204
"""

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from scipy.spatial.distance import cdist

from federated_umap.task import (
    get_model,
    load_data,
    load_deployment_data,
    set_model_params,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_p = X_train.astype(np.float64)
        self.labels = y_train

    def fit(self, parameters, config):
        # Algorithm 1, step 5: initialise local Y from the server broadcast.
        set_model_params(self.model, parameters)

        # Algorithm 2, step 2: compute D_{X_p, Y} using the *pre-update* Y.
        # This distance matrix is what the server needs for Nyström reconstruction
        # (eq. 21).  It must be computed before the local gradient step so it
        # reflects the same Y that was broadcast, keeping all clients consistent.
        D_Xp_Y = self._compute_distance_to_landmarks()  # (n_p, n_y)

        # Algorithm 1, line 8: local gradient descent step on f_p(Y) = MMD(X_p, Y).
        self.model.Y = self.model.forward(self.X_p)

        # Algorithm 1, step 11: upload Y_p and D_{X_p,Y} to the server.
        return (
            [self.model.get_Y(), D_Xp_Y, self.labels.astype(np.float64)],
            len(self.X_p),
            {},
        )

    def evaluate(self, parameters, config):
        return 0.5, len(self.X_p), {"accuracy": 0.0}

    def _compute_distance_to_landmarks(self):
        """
        Algorithm 2, step 2 / paper eq. 21 (one row-block of B):
            D_{X_p, Y} ∈ R^{n_p × n_y}
        Euclidean distances from each local data point to each landmark.
        Raw data X_p never leaves the client — only this distance matrix is sent.
        """
        X_flat = self.X_p.reshape(self.X_p.shape[0], -1)
        Y_flat = self.model.Y.reshape(self.model.n_y, -1)
        return cdist(X_flat, Y_flat, metric="euclidean").astype(np.float64)


def client_fn(context: Context):
    dataset_name = context.run_config.get("dataset", "ylecun/mnist")
    feature_column = context.run_config.get("feature-column", "image")
    label_column = context.run_config.get("label-column", "label")

    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        # Simulation Engine: partition the dataset on the fly with flwr_datasets
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        X, y = load_data(
            partition_id, num_partitions, dataset_name, feature_column, label_column
        )
    else:
        # Deployment Engine: load a pre-partitioned dataset from local disk.
        # Generate demo data with:
        #   flwr-datasets create <dataset> --num-partitions N --out-dir <out>
        # Then pass the partition directory as data-path in node_config.
        data_path = context.node_config["data-path"]
        X, y = load_deployment_data(data_path, feature_column, label_column)

    local_epochs = context.run_config["local-epochs"]
    params = {
        "shape": (X.shape[1],),  # inferred from loaded data
        "n-y": context.run_config["n-y"],
    }
    model = get_model(params, local_epochs)

    return FlowerClient(model, X, y).to_client()


app = ClientApp(client_fn=client_fn)
