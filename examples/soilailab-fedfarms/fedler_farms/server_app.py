from __future__ import annotations

import os
from typing import Any, Dict

import flwr as fl
import numpy as np
import pandas as pd
import torch

from .dataset import make_federated_dataset, load_global_test_split
from .model import create_model
from .task import ensure_dir, from_weights, rc_get, rmse, r2, rpiq, to_weights


def _feature_cols(rc: Dict[str, Any]) -> list[str]:
    n = int(rc_get(rc, "num-features", default=10))
    return [f"X{i}" for i in range(1, n + 1)]


def _target_cols(rc: Dict[str, Any]) -> list[str]:
    return list(rc_get(rc, "targets", default="Clay_gkg_filtered,C_gkg_filtered").split(","))


def server_fn(context: fl.common.Context) -> fl.server.ServerAppComponents:
    rc = context.run_config

    num_rounds = int(rc_get(rc, "num-server-rounds", "num-rounds", default=5))
    num_clients = int(rc_get(rc, "num-clients", default=3))

    metrics_dir = str(rc_get(rc, "metrics-dir", default="outputs/metrics_demo"))
    ensure_dir(metrics_dir)

    hf_dataset = str(rc_get(rc, "hf-dataset"))
    partition_by = str(rc_get(rc, "partition-by", default="farm"))

    feature_cols = _feature_cols(rc)
    target_cols = _target_cols(rc)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(
        input_shape=(len(feature_cols), 1),
        output_dim=len(target_cols),
        conv1_filters=int(rc_get(rc, "conv1-filters", default=64)),
        conv2_filters=int(rc_get(rc, "conv2-filters", default=64)),
        kernel_size=int(rc_get(rc, "kernel-size", default=5)),
        use_pooling=bool(rc_get(rc, "use-pooling", default=False)),
    ).to(device)

    initial_parameters = fl.common.ndarrays_to_parameters(to_weights(model))

    # Load centralized test split
    fds = make_federated_dataset(hf_dataset=hf_dataset, partition_by=partition_by)
    Xg, yg = load_global_test_split(fds, feature_cols, target_cols)
    y_true_np = yg.values

    best_loss = float("inf")
    best_round = -1

    def evaluate_fn(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ):
        nonlocal best_loss, best_round

        # Convert Parameters -> NDArrays if needed
        if hasattr(parameters, "tensors"):
            parameters = fl.common.parameters_to_ndarrays(parameters)

        from_weights(model, parameters)
        y_pred = _predict_on_df(model, Xg, device)

        # loss = mean MSE across targets
        mse_vals = np.mean((y_true_np - y_pred) ** 2, axis=0)
        loss = float(np.mean(mse_vals))

        r2v = r2(y_true_np, y_pred)
        rmsev = rmse(y_true_np, y_pred)
        rpiqv = rpiq(y_true_np, y_pred)

        metrics = {"round": int(server_round), "average_loss": float(loss)}
        for i, name in enumerate(target_cols):
            metrics[f"R2-{name}"] = float(r2v[i])
            metrics[f"RMSE-{name}"] = float(rmsev[i])
            metrics[f"RPIQ-{name}"] = float(rpiqv[i])

        # save best predictions
        if loss < best_loss:
            best_loss = loss
            best_round = server_round
            df_pred = pd.DataFrame(y_pred, columns=[f"Pred_{c}" for c in target_cols])
            df_true = yg.reset_index(drop=True).add_prefix("True_")
            df_out = pd.concat([df_true, df_pred], axis=1)
            df_out["best_round"] = best_round
            df_out.to_csv(os.path.join(metrics_dir, "obs_pred_best.csv"), index=False)

        # append round metrics
        out_csv = os.path.join(metrics_dir, "server_metrics.csv")
        df_row = pd.DataFrame([metrics])
        if os.path.exists(out_csv):
            df_row.to_csv(out_csv, mode="a", header=False, index=False)
        else:
            df_row.to_csv(out_csv, mode="w", header=True, index=False)

        print(f"[Server] Round {server_round} loss={loss:.6f} best_round={best_round}")
        return loss, metrics

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        evaluate_fn=evaluate_fn,
    )

    server_config = fl.server.ServerConfig(num_rounds=num_rounds)
    return fl.server.ServerAppComponents(strategy=strategy, config=server_config)


def _predict_on_df(model: torch.nn.Module, X: pd.DataFrame, device: torch.device) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        out = model(Xt)
    return out.detach().cpu().numpy()


app = fl.server.ServerApp(server_fn=server_fn)