"""Federated UMAP — server-side strategy and Nyström reconstruction.

Implements the server component from:
    "Federated t-SNE and UMAP for Distributed Data Visualization"
    Dong Qiao*, Xinxian Ma*, Jicong Fan†
    The 39th AAAI Conference on Artificial Intelligence (AAAI-25), 2025
    * equal contribution  † corresponding author
    https://ojs.aaai.org/index.php/AAAI/article/view/34204
"""

import io

import numpy as np
import umap
import wandb
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from scipy.spatial.distance import cdist

from federated_umap.task import get_model, get_model_params


def nystrom_distance_matrix(B, W, rank=None):
    """Algorithm 2, step 3 — estimate the full pairwise distance matrix via the Nyström
    method (paper eq. 22):

        D̂_{X,X} = B · W†_k · B^T

    where (paper eq. 21):
        B  = [D_{X1,Y}; D_{X2,Y}; ...; D_{XP,Y}]  (n_x, n_y)  stacked client matrices
        W  = D_{Y,Y}                                (n_y, n_y)  landmark distance matrix
        W†_k = rank-k pseudoinverse of W            (n_y, n_y)

    The rank-k truncation follows the theoretical motivation in paper §4 /
    Theorem 2 (Nyström error bounds, citing Drineas & Mahoney 2005): keeping
    only the top-k singular values suppresses noise while controlling the
    approximation error ‖D_{X,X} - D̂_{X,X}‖.

    Args:
        B:    (n_x, n_y) stacked client distance matrices
        W:    (n_y, n_y) landmark-to-landmark distance matrix
        rank: rank k for approximation (None = full rank pseudoinverse)

    Returns:
        D̂: (n_x, n_x) estimated distance matrix (non-negative, symmetric)
    """
    n_y = W.shape[0]
    k = rank if rank is not None else n_y

    # Regularise to avoid singularity
    W_reg = W + 1e-6 * np.eye(n_y)

    # Rank-k pseudoinverse via truncated SVD.
    # Use a *relative* threshold so the cutoff scales with the data magnitude
    # (absolute thresholds like 1e-10 are too small when W entries are O(100),
    # causing divide-by-zero / overflow in the subsequent matmuls).
    U, s, Vt = np.linalg.svd(W_reg, full_matrices=False)
    s_k = s[:k].copy()
    threshold = max(s_k[0] * 1e-8, 1e-12) if s_k[0] > 0 else 1e-12
    s_k = np.maximum(s_k, threshold)

    # BLAS reports spurious over/underflow warnings on intermediate products
    # even when the final result is finite — suppress them.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        W_pinv_k = (Vt[:k].T * (1.0 / s_k)) @ U[:, :k].T  # (n_y, n_y)
        D_hat = B @ W_pinv_k @ B.T  # (n_x, n_x)

    # Distance matrices must be non-negative and symmetric
    D_hat = np.maximum(D_hat, 0.0)
    D_hat = 0.5 * (D_hat + D_hat.T)
    np.fill_diagonal(D_hat, 0.0)

    return D_hat


def _make_umap(metric="euclidean"):
    return umap.UMAP(
        n_components=2,
        metric=metric,
        n_neighbors=15,
        min_dist=0.1,
        random_state=42,
    )


def _make_scatter_table(embedding, labels):
    """Build a wandb.Table with (x, y, label) rows from an (n, 2) embedding."""
    table = wandb.Table(columns=["x", "y", "label"])
    for (x, y), label in zip(embedding, labels):
        table.add_data(float(x), float(y), int(label))
    return table


class FedUMAPStrategy(FedAvg):
    """Flower strategy implementing Fed-UMAP (paper Algorithm 2).

    Each round:
      - Receives Y_p and D_{X_p,Y} from every client (Algorithm 2, step 2).
      - Aggregates Y via weighted FedAvg (Algorithm 1, step 13, eq. 15).
      - Broadcasts the new global Y back to clients (Algorithm 1, step 15).

    Final round only:
      - Assembles B = [D_{X1,Y}; ...; D_{XP,Y}] (eq. 21).
      - Reconstructs D̂_{X,X} via Nyström (eq. 22) — Algorithm 2, step 3.
      - Runs UMAP on D̂_{X,X} (Algorithm 2, step 4).
      - Also runs UMAP on K_XY kernel features as an alternative view.
      - Logs both embeddings and the final Y artifact to W&B.

    After the final round, results are accessible via:
        strategy.embedding        (n_x, 2)  UMAP coordinates (Nyström, eq. 22)
        strategy.embedding_labels (n_x,)    class labels
    """

    def __init__(
        self, num_rounds, n_y, umap_rank=50, umap_max_samples=10_000, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_rounds = num_rounds
        self.n_y = n_y
        self.umap_rank = umap_rank
        self.umap_max_samples = umap_max_samples

        # Accumulated across all rounds; overwritten each round
        self._last_B = None  # stacked D_{Xp,Y}  (n_x, n_y)
        self._last_labels = None  # all labels         (n_x,)
        self._last_Y = None  # aggregated Y       (n_y, d)

        # Final results, available after the last round
        self.embedding = None  # (n_x, 2)  Nyström embedding
        self.embedding_labels = None  # (n_x,)

    def aggregate_fit(self, rnd, results, failures):
        if not results:
            return None, {}

        # ----------------------------------------------------------------
        # Unpack the three arrays each client returns (Algorithm 2, step 2):
        #   [0] Y_p        (n_y, d)   — updated landmarks (Algorithm 1, step 11)
        #   [1] D_{Xp,Y}   (n_p, n_y) — client-to-landmark distances (eq. 21 row block)
        #   [2] labels     (n_p,)     — class labels for final visualisation
        # ----------------------------------------------------------------
        Y_updates, D_list, label_list = [], [], []

        for _proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            Y_p = arrays[0]
            D_Xp_Y = arrays[1]
            labels_p = arrays[2].astype(int)

            Y_updates.append((fit_res.num_examples, Y_p))
            D_list.append(D_Xp_Y)
            label_list.append(labels_p)

        # FedAvg over Y — Algorithm 1, step 13 / eq. 15:
        #   Y^s = (1/n_x) Σ_p n_p · Y_p^s   (weighted average)
        total = sum(n for n, _ in Y_updates)
        Y_agg = sum(n / total * Y for n, Y in Y_updates)

        # Assemble B per eq. 21: B = [D_{X1,Y}; ...; D_{XP,Y}]  (n_x, n_y)
        B = np.vstack(D_list).astype(np.float64)
        labels_all = np.concatenate(label_list)

        self._last_B = B
        self._last_labels = labels_all
        self._last_Y = Y_agg

        print(f"[Round {rnd}] Y aggregated | B shape: {B.shape}")

        wandb.log(
            {
                "train/n_clients": len(results),
                "train/n_samples": int(B.shape[0]),
                "train/Y_norm": float(np.linalg.norm(Y_agg)),
                "train/Y_mean": float(Y_agg.mean()),
            },
            step=rnd,
        )

        # On final round: run full Nystrom + UMAP
        if rnd == self.num_rounds:
            self._run_final_umap(Y_agg, B, labels_all, rnd)

        return ndarrays_to_parameters([Y_agg]), {}

    def _run_final_umap(self, Y, B, labels, rnd):
        """
        Compute two federated UMAP embeddings for comparison and log to W&B:
          1. K_XY features — UMAP on Gaussian kernel of B
          2. Nyström D̂    — UMAP on Nyström-reconstructed distance matrix

        Raw client data never reaches the server — both embeddings are derived
        entirely from B (distances to landmarks) and Y (the landmarks themselves).

        Subsampling is applied *before* any computation so that the Nyström
        step never materialises a matrix larger than (umap_max_samples**2),
        avoiding the ~28 GB allocation that occurs with 60k MNIST samples for example.
        The same index is reused for both embeddings for consistency.
        """
        n_total = B.shape[0]

        if n_total > self.umap_max_samples:
            rng = np.random.default_rng(42)
            idx = np.sort(rng.choice(n_total, self.umap_max_samples, replace=False))
            print(
                f"[Round {rnd}] Subsampling {self.umap_max_samples}/{n_total} "
                f"points for UMAP visualisation"
            )
        else:
            idx = np.arange(n_total)

        B_sub = B[idx]
        labels_sub = labels[idx]

        # ── Nyström reconstruction — Algorithm 2, step 3, eq. 22 ─────────
        # D̂_{X,X} = B · W†_k · B^T  where W = D_{Y,Y}
        # D_hat is (umap_max_samples × umap_max_samples) — manageable after subsampling.
        print(f"[Round {rnd}] Running Nystrom reconstruction on B_sub={B_sub.shape}...")
        W = cdist(Y, Y, metric="euclidean").astype(np.float64)
        rank = min(self.umap_rank, W.shape[0])
        D_hat = nystrom_distance_matrix(B_sub, W, rank=rank)
        print(
            f"[Round {rnd}] D̂ shape: {D_hat.shape}, "
            f"range: [{D_hat.min():.3f}, {D_hat.max():.3f}]"
        )
        # Algorithm 2, step 4: run UMAP on the reconstructed distance matrix.
        print(f"[Round {rnd}] Running UMAP (Nyström)...")
        emb_nystrom = _make_umap(metric="precomputed").fit_transform(D_hat)

        print(f"[Round {rnd}] Running UMAP (K_XY features)...")
        pos_vals = B_sub[B_sub > 0]
        med = float(np.median(pos_vals)) if pos_vals.size else 1.0
        gamma = 1.0 / (2.0 * med**2) if med > 0 else 1.0
        K_XY_sub = np.exp(-gamma * B_sub**2)
        K_dist = cdist(K_XY_sub, K_XY_sub, metric="euclidean").astype(np.float64)
        emb_kxy = _make_umap(metric="precomputed").fit_transform(K_dist)

        self.embedding = emb_nystrom
        self.embedding_labels = labels_sub
        print(f"[Round {rnd}] All embeddings complete.")

        self._log_to_wandb(Y, labels_sub, emb_kxy, emb_nystrom, rnd)

    def _log_to_wandb(self, Y, labels, emb_kxy, emb_nystrom, rnd):
        """Log two federated UMAP scatter plots to W&B:

          - embedding/kxy_features  — UMAP on K_XY kernel features
          - embedding/nystrom       — UMAP on Nyström-reconstructed D̂
        Each is a wandb.Table (for custom charts) + wandb.plot.scatter
        (for instant preview).  In the W&B UI open any table and choose
        Add visualization → Scatter plot, colour = label.
        """
        plots = {
            "nystrom": (emb_nystrom, "Fed-UMAP via Nyström D̂"),
            "kxy_features": (emb_kxy, "Fed-UMAP via K_XY features"),
        }

        log_dict = {}
        for key, (emb, title) in plots.items():
            table = _make_scatter_table(emb, labels)
            log_dict[f"embedding/{key}"] = wandb.plot.scatter(
                table, "x", "y", title=f"{title} — round {rnd}"
            )
            log_dict[f"embedding/{key}_table"] = table

        wandb.log(log_dict, step=rnd)

        # ── Y landmarks artifact (the "model weights") ───────────────────
        buf = io.BytesIO()
        np.save(buf, Y)
        buf.seek(0)

        artifact = wandb.Artifact(
            name="landmarks",
            type="model",
            description=f"Final aggregated Y landmarks — n_y={Y.shape[0]}, d={Y.shape[1]}",
            metadata={"n_y": Y.shape[0], "feature_dim": Y.shape[1], "round": rnd},
        )
        with artifact.new_file("Y_final.npy", mode="wb") as f:
            f.write(buf.read())
        wandb.log_artifact(artifact)

        print(
            f"[Round {rnd}] W&B: 2-panel embedding comparison + landmark artifact logged."
        )
        wandb.finish()

    def evaluate(self, server_round, parameters):
        return None


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    n_y = context.run_config["n-y"]
    feature_dim = context.run_config["feature-dim"]

    wandb.init(
        project=context.run_config.get("wandb-project", "federated-umap"),
        config=dict(context.run_config),
        tags=[context.run_config.get("dataset", "unknown")],
    )

    params = {
        "shape": (feature_dim,),
        "n-y": n_y,
    }
    model = get_model(params, local_epochs=1)
    initial_parameters = ndarrays_to_parameters(get_model_params(model))

    strategy = FedUMAPStrategy(
        num_rounds=num_rounds,
        n_y=n_y,
        umap_rank=context.run_config.get("umap-rank", 50),
        umap_max_samples=context.run_config.get("umap-max-samples", 10_000),
        fraction_fit=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
