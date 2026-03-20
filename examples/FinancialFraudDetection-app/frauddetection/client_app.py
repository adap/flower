"""frauddetection: Flower ClientApp for federated XGBoost fraud detection."""

import os

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from frauddetection.task import (
    deserialize_model,
    evaluate_xgboost,
    load_local_data,
    load_sim_data,
    model_bytes_to_numpy,
    numpy_to_model_bytes,
    serialize_model,
    train_xgboost,
)

# Flower ClientApp
app = ClientApp()


def _bundled_csv(context: Context) -> str:
    """Resolve the data CSV path.

    Priority:
    1. ``data-csv`` key in run_config (absolute or relative to CWD).
    2. Default relative path ``data/preprocessed_Ethereum_cleaned_v2.csv``.
    """
    csv_path = context.run_config.get(
        "data-csv", "data/preprocessed_Ethereum_cleaned_v2.csv"
    )
    return str(csv_path)


def _load_data(context: Context):
    """Return (X_train, X_test, y_train, y_test) for the current node.

    Simulation engine   — uses ``partition-id`` / ``num-partitions`` from
                          node_config to slice the bundled CSV on the fly.
    Deployment engine   — reads a pre-split CSV from the ``data-path`` key
                          in node_config.
    """
    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        # Simulation mode
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        data_csv = _bundled_csv(context)
        return load_sim_data(partition_id, num_partitions, data_csv)
    else:
        # Deployment mode
        data_path = context.node_config["data-path"]
        return load_local_data(data_path)


# ──────────────────────────────────────────────
# Train handler
# ──────────────────────────────────────────────

@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train a local XGBoost model and return the serialized booster."""

    local_epochs: int = int(context.run_config.get("local-epochs", 50))

    # Load partition data
    X_train, X_test, y_train, y_test = _load_data(context)

    # Train XGBoost locally (from scratch each round — FedXGBBagging collects
    # all round models for the final ensemble, so no warm-starting needed)
    booster = train_xgboost(X_train, y_train, local_epochs=local_epochs)

    # Evaluate on local hold-out set
    acc, auc = evaluate_xgboost(booster, X_test, y_test)

    # Encode model as uint8 numpy array for transmission
    model_bytes = serialize_model(booster)
    model_array = model_bytes_to_numpy(model_bytes)

    model_record = ArrayRecord({"model_bytes": model_array})
    metric_record = MetricRecord(
        {
            "train_acc": float(acc),
            "train_auc": float(auc),
            "num_examples": float(len(y_train)),
        }
    )
    content = RecordDict({"model": model_record, "metrics": metric_record})

    node_id = context.node_id if hasattr(context, "node_id") else "?"
    print(
        f"[Client {node_id}] train done — "
        f"acc={acc:.4f}  auc={auc:.4f}  examples={len(y_train)}"
    )
    return Message(content=content, reply_to=msg)


# ──────────────────────────────────────────────
# Evaluate handler
# ──────────────────────────────────────────────

@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global representative model on local test data.

    The server sends one of the collected client models (or the first model
    of the ensemble) for distributed evaluation.  This gives a per-partition
    view of generalisation quality.
    """

    # Receive model bytes from server message
    model_record: ArrayRecord = msg.content["model"]
    model_bytes = numpy_to_model_bytes(model_record["model_bytes"])
    booster = deserialize_model(model_bytes)

    # Load local test split
    _, X_test, _, y_test = _load_data(context)

    acc, auc = evaluate_xgboost(booster, X_test, y_test)

    metric_record = MetricRecord(
        {
            "eval_acc": float(acc),
            "eval_auc": float(auc),
            "num_examples": float(len(y_test)),
        }
    )
    content = RecordDict({"metrics": metric_record})

    node_id = context.node_id if hasattr(context, "node_id") else "?"
    print(
        f"[Client {node_id}] eval done — "
        f"acc={acc:.4f}  auc={auc:.4f}  examples={len(y_test)}"
    )
    return Message(content=content, reply_to=msg)
