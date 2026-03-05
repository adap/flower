from __future__ import annotations

from typing import Any, Dict

import flwr as fl
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .dataset import load_client_partition, make_federated_dataset
from .model import create_model
from .task import (
    TrainConfig,
    ensure_dir,
    from_weights,
    predict,
    r2,
    rc_get,
    rmse,
    rpiq,
    to_weights,
    train_local,
)


def _feature_cols(rc: Dict[str, Any]) -> list[str]:
    n = int(rc_get(rc, "num-features", default=10))
    return [f"X{i}" for i in range(1, n + 1)]


def _target_cols(rc: Dict[str, Any]) -> list[str]:
    return list(
        rc_get(rc, "targets", default="Clay_gkg_filtered,C_gkg_filtered").split(",")
    )


class FarmClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: int,
        model: torch.nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        y_val_np: np.ndarray,
        target_cols: list[str],
        train_cfg: TrainConfig,
        outputs_dir: str,
    ):
        self.cid = cid
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.y_val_np = y_val_np
        self.target_cols = target_cols
        self.train_cfg = train_cfg
        self.outputs_dir = outputs_dir
        self.round = 0

    def get_parameters(self, config):
        return to_weights(self.model)

    def fit(self, parameters, config):
        from_weights(self.model, parameters)

        best_val = train_local(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            cfg=self.train_cfg,
            device=self.device,
        )

        y_pred = predict(self.model, self.val_loader, self.device)
        y_true = self.y_val_np

        r2v = r2(y_true, y_pred)
        rmsev = rmse(y_true, y_pred)
        rpiqv = rpiq(y_true, y_pred)

        metrics = {
            "client_id": int(self.cid),
            "round": int(self.round),
            "val_loss": float(best_val),
        }
        for i, name in enumerate(self.target_cols):
            metrics[f"R2-{name}"] = float(r2v[i])
            metrics[f"RMSE-{name}"] = float(rmsev[i])
            metrics[f"RPIQ-{name}"] = float(rpiqv[i])

        self.round += 1
        return to_weights(self.model), len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        from_weights(self.model, parameters)

        y_pred = predict(self.model, self.val_loader, self.device)
        y_true = self.y_val_np

        r2v = r2(y_true, y_pred)
        rmsev = rmse(y_true, y_pred)
        rpiqv = rpiq(y_true, y_pred)

        # loss = mean MSE across targets
        mse_vals = np.mean((y_true - y_pred) ** 2, axis=0)
        loss = float(np.mean(mse_vals))

        metrics = {"client_id": int(self.cid), "round": int(self.round)}
        for i, name in enumerate(self.target_cols):
            metrics[f"R2-{name}"] = float(r2v[i])
            metrics[f"RMSE-{name}"] = float(rmsev[i])
            metrics[f"RPIQ-{name}"] = float(rpiqv[i])

        return loss, len(self.val_loader.dataset), metrics


def client_fn(context: fl.common.Context) -> fl.client.Client:
    rc = context.run_config

    # robust partition id
    pid = context.node_config.get("partition-id", None)

    num_clients = int(rc_get(rc, "num-clients", default=1))

    if pid is not None:
        partition_id = int(pid)
    else:
        # Try to get a stable node identifier from the context/node_config
        node_id = getattr(context, "node_id", None) or context.node_config.get(
            "node-id", None
        )

        if node_id is None:
            partition_id = 0
        else:
            # Deterministic mapping to [0, num_clients-1]
            partition_id = abs(hash(str(node_id))) % max(num_clients, 1)

    hf_dataset = str(rc_get(rc, "hf-dataset"))
    partition_by = str(rc_get(rc, "partition-by", default="farm"))

    val_ratio = float(rc_get(rc, "val-ratio", default=0.3))
    seed = int(rc_get(rc, "seed", default=42))

    batch_size = int(rc_get(rc, "batch-size", default=8))
    local_epochs = int(rc_get(rc, "local-epochs", default=3))
    lr = float(rc_get(rc, "learning-rate", default=1e-3))
    patience = int(rc_get(rc, "early-stop-patience", default=10))

    outputs_dir = str(rc_get(rc, "metrics-dir", default="outputs/metrics_demo"))
    ensure_dir(outputs_dir)

    feature_cols = _feature_cols(rc)
    target_cols = _target_cols(rc)

    # Build federated dataset + load this client's partition
    fds = make_federated_dataset(hf_dataset=hf_dataset, partition_by=partition_by)
    X, y = load_client_partition(fds, partition_id, feature_cols, target_cols)

    # split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=seed, shuffle=True
    )

    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(
        input_shape=(len(feature_cols), 1),
        output_dim=len(target_cols),
        conv1_filters=int(rc_get(rc, "conv1-filters", default=64)),
        conv2_filters=int(rc_get(rc, "conv2-filters", default=64)),
        kernel_size=int(rc_get(rc, "kernel-size", default=5)),
        use_pooling=bool(rc_get(rc, "use-pooling", default=False)),
    )

    train_cfg = TrainConfig(
        batch_size=batch_size,
        local_epochs=local_epochs,
        learning_rate=lr,
        early_stop_patience=patience,
    )

    return FarmClient(
        cid=partition_id,
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        y_val_np=y_val.values,
        target_cols=target_cols,
        train_cfg=train_cfg,
        outputs_dir=outputs_dir,
    ).to_client()


app = fl.client.ClientApp(client_fn=client_fn)
