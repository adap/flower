"""
Federated UMAP — client-side model and data utilities.

Implements the local optimisation component from:
    "Federated t-SNE and UMAP for Distributed Data Visualization"
    Dong Qiao*, Xinxian Ma*, Jicong Fan†
    The 39th AAAI Conference on Artificial Intelligence (AAAI-25), 2025
    * equal contribution  † corresponding author
    https://ojs.aaai.org/index.php/AAAI/article/view/34204
"""

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from scipy.spatial.distance import pdist

_fds_cache: dict = {}  # Cache FederatedDataset by dataset name


class FedMMDClient:
    """
    Local landmark optimiser for one federated client (paper Algorithm 1 — FedDL).

    Each client holds a local copy of the shared landmark matrix Y ∈ R^{n_y × d}
    and minimises f_p(Y) = MMD(X_p, Y) (paper eq. 9/10) via gradient descent
    (paper eq. 13) so that Y captures the distribution of its local data X_p.

    After Q local steps the updated Y_p is uploaded to the server for FedAvg
    aggregation (Algorithm 1, steps 11 & 13).  The server broadcasts the new
    global Y back (step 15) and the cycle repeats for S rounds.
    """

    def __init__(self, params: dict):
        self.n_y = params.get("n-y", 110)
        self.shape = params["shape"]
        feature_dim = int(np.prod(params["shape"]))

        self.Y = self._init_Y(self.n_y, feature_dim).astype(np.float64)

        # gamma is set from data on first forward() call, fixed thereafter
        self.gamma = params.get("gamma", None)  # None = compute from X
        self._step_size = params.get("step_size", 0.01)

    def forward(self, X):
        """
        One local gradient descent step — Algorithm 1, line 8:
            Y_p^{s,t} = Y_p^{s,t-1} - η · ∇f_p(Y_p^{s,t-1})
        X: (n_p, d)  ->  returns updated Y: (n_y, d)
        """
        X = self._to_2d_f64(X)
        Y = self._to_2d_f64(self.Y)

        # Set gamma from DATA scale on first call, then freeze it.
        # Gamma must not track Y — if it does, shrinking gamma kills gradients.
        if self.gamma is None:
            self.gamma = self._gamma_from_data(X)
            print(f"[FedMMDClient] gamma set to {self.gamma:.6f} from data median")

        gradient = self._compute_gradient(X, Y, self.gamma)

        # Fixed step size — simpler and more stable than adaptive Lipschitz.
        Y_new = Y - self._step_size * gradient

        if not np.isfinite(Y_new).all():
            print("[FedMMDClient] WARNING: non-finite Y, keeping previous")
            Y_new = Y

        self.Y = Y_new
        return self.Y

    def update_Y(self, new_Y):
        """
        Receive aggregated Y from server.
        Algorithm 1, step 1 (initial broadcast) and step 15 (per-round broadcast).
        """
        self.Y = np.array(new_Y, dtype=np.float64)

    def get_Y(self):
        """
        Return Y_p for upload to server.
        Algorithm 1, step 11: upload Y_p^s to the server for averaging.
        """
        return self.Y.copy()

    def get_gradient(self, X):
        """
        Return ∇f_p(Y) for gradient-based server aggregation.
        Used in the alternative update rule (paper eq. 16):
            Y^s ← Y^{s-1} - η' × (1/P) Σ_p ∇f_p(Y_p^s)
        """
        X = self._to_2d_f64(X)
        Y = self._to_2d_f64(self.Y)
        if self.gamma is None:
            self.gamma = self._gamma_from_data(X)
        return self._compute_gradient(X, Y, self.gamma)

    def _compute_gradient(self, X, Y, gamma):
        """
        Gradient of f_p(Y) = MMD(X_p, Y) w.r.t. Y — paper eq. (13).
        Returns shape (n_y, d).

        From eq. (13) (using row-vector convention, i.e. X is (n_p, d)):
            ∇f_p(Y) = -4γ/(n_p·n_y) · [K_{XY}^T X - diag(K_{XY}^T 1) Y]   ← term1
                     + 4γ/(n_y(n_y-1)) · [K_{YY} Y - diag(K_{YY}^T 1) Y]   ← term2

        Intuition:
          term1 (attraction): pulls each landmark toward the data it is closest to.
          term2 (repulsion):  pushes landmarks apart so they cover the data support.
        """
        n_p = X.shape[0]
        n_y = Y.shape[0]

        K_XY = self._gaussian_kernel(X, Y, gamma)  # (n_p, n_y)
        K_YY = self._gaussian_kernel(Y, Y, gamma)  # (n_y, n_y)

        # Term 1: attraction — pulls landmarks toward data
        col_sums_XY = K_XY.sum(axis=0)  # (n_y,)
        term1 = (-4.0 * gamma / (n_p * n_y)) * (
            K_XY.T @ X - col_sums_XY[:, None] * Y  # (n_y, d)  # (n_y, d)
        )

        # Term 2: repulsion — spreads landmarks apart
        if n_y > 1:
            col_sums_YY = K_YY.sum(axis=0)  # (n_y,)
            term2 = (4.0 * gamma / (n_y * (n_y - 1))) * (
                K_YY @ Y - col_sums_YY[:, None] * Y  # (n_y, d)  # (n_y, d)
            )
        else:
            term2 = np.zeros_like(Y)

        return term1 + term2  # (n_y, d)

    def _gaussian_kernel(self, X, Y, gamma):
        """
        Gaussian kernel matrix K_{X,Y} with bandwidth γ (paper §3.1):
            k(x_i, y_j) = exp(-γ · ‖x_i - y_j‖^2)
            K_{X,Y} = exp(-γ · D^2_{X,Y})
        where D^2_{X,Y} is the squared pairwise distance matrix (eq. after eq. 10).
        Output is clipped to [-500, 0] before exp for numerical stability.
        """
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        Y_sq = np.sum(Y**2, axis=1, keepdims=True).T
        # BLAS can emit spurious over/underflow warnings on the cross-term
        # when Y drifts during optimisation — suppress and replace any
        # non-finite intermediates with 0 before clamping sq_dist.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            cross = X @ Y.T
        cross = np.nan_to_num(cross, nan=0.0, posinf=0.0, neginf=0.0)
        sq_dist = np.maximum(X_sq - 2.0 * cross + Y_sq, 0.0)
        return np.exp(np.clip(-gamma * sq_dist, -500.0, 0.0))

    def _gamma_from_data(self, X):
        """
        Compute gamma from a subsample of X using the median heuristic.
        This anchors gamma to the data scale and keeps it stable.
        """
        sample = X[:500] if len(X) > 500 else X
        dists = pdist(sample)
        med = np.median(dists)
        return 1.0 / (2.0 * med**2) if med > 0 else 1.0

    @staticmethod
    def _to_2d_f64(arr):
        arr = np.array(arr, dtype=np.float64)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr

    def _init_Y(self, n_y, feature_dim):
        """Initialize Y uniformly in [0, 1]."""
        return np.random.uniform(0.0, 1.0, size=(n_y, feature_dim))


def load_data(
    partition_id: int,
    num_partitions: int,
    dataset_name: str = "ylecun/mnist",
    feature_column: str = "image",
    label_column: str = "label",
):
    """Load a partition of any HuggingFace image dataset, normalized to [0, 1] as float64."""
    global _fds_cache
    if dataset_name not in _fds_cache:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        _fds_cache[dataset_name] = FederatedDataset(
            dataset=dataset_name,
            partitioners={"train": partitioner},
        )
    dataset = (
        _fds_cache[dataset_name]
        .load_partition(partition_id, "train")
        .with_format("numpy")
    )
    images = dataset[feature_column]
    X = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0
    y = dataset[label_column]
    return X, y


def load_deployment_data(
    data_path: str,
    feature_column: str = "image",
    label_column: str = "label",
):
    """
    Load a pre-partitioned dataset from disk (Deployment Engine mode).

    Expects a directory created by:
        flwr-datasets create <dataset> --num-partitions N --out-dir <out>
    which saves each partition as a HuggingFace dataset shard under
    <out>/partition_<i>/.  Pass the specific partition directory as data_path.

    Returns X: (n, d) float64 in [0, 1] and y: (n,) labels.
    """
    from datasets import load_from_disk

    dataset = load_from_disk(data_path).with_format("numpy")
    images = dataset[feature_column]
    X = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0
    y = dataset[label_column]
    return X, y


def get_model(params: dict, local_epochs: int):
    return FedMMDClient(params)


def get_model_params(model):
    return [model.get_Y()]


def set_model_params(model, params):
    model.update_Y(params[0])
    return model
