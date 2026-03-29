"""frauddetection: Flower ServerApp for federated XGBoost fraud detection.

Federation strategy
-------------------
Each training round, every client:
  1. Trains a fresh XGBoost model on its local partition.
  2. Returns the serialised booster (JSON bytes packed as uint8 ndarray).

The server collects all models across all rounds and builds a
**FedXGBBagging** ensemble for the final inference step.

After all rounds an optional centralised evaluation is performed on the
last 20 % of the bundled CSV (server-side held-out set).

The final ensemble model files are written to ``./final_ensemble/``.
"""

import os
import shutil
import tempfile

import numpy as np
import pandas as pd
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, RecordDict
from flwr.serverapp import Grid, ServerApp

from frauddetection.fed_xgb_bagging import FedXGBBagging
from frauddetection.task import LABEL_COL, CAT_COLS, numpy_to_model_bytes, preprocess_df

# Flower ServerApp
app = ServerApp()


# ──────────────────────────────────────────────
# Helper: central test data
# ──────────────────────────────────────────────

def _bundled_csv() -> str:
    return os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "preprocessed_Ethereum_cleaned_v2.csv",
    )


def _load_central_test(csv_path: str, test_fraction: float = 0.2) -> tuple:
    """Load the last ``test_fraction`` of the CSV as a held-out test set."""
    df = pd.read_csv(csv_path)
    n = len(df)
    test_df = df.iloc[int(n * (1 - test_fraction)):].reset_index(drop=True)
    X, y = preprocess_df(test_df)
    return X, y


# ──────────────────────────────────────────────
# Server main loop
# ──────────────────────────────────────────────

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Orchestrate federated XGBoost training and build the final ensemble."""

    num_rounds: int = int(context.run_config.get("num-server-rounds", 3))
    fraction_evaluate: float = float(
        context.run_config.get("fraction-evaluate", 1.0)
    )

    node_ids = list(grid.get_node_ids())
    n_clients = len(node_ids)
    print(
        f"\n[Server] Starting FedXGBBagging federation"
        f" — clients={n_clients}  rounds={num_rounds}"
    )

    # Temporary directory to store serialised booster files
    tmp_dir = tempfile.mkdtemp(prefix="fed_xgb_")
    all_model_paths: list[str] = []

    try:
        # ── Training rounds ──────────────────────────────────────────────
        for server_round in range(1, num_rounds + 1):
            print(f"\n[Server] ── Round {server_round}/{num_rounds} ──")

            # Build one "train" message per client
            train_messages = [
                grid.create_message(
                    content=RecordDict(
                        {
                            "config": ConfigRecord(
                                {"round": server_round, "num_rounds": num_rounds}
                            )
                        }
                    ),
                    message_type="train",
                    dst_node_id=nid,
                    group_id=str(server_round),
                )
                for nid in node_ids
            ]

            replies = list(grid.send_and_receive(train_messages))

            # Collect and persist each client's model
            round_paths: list[str] = []
            for i, reply in enumerate(replies):
                model_record: ArrayRecord = reply.content["model"]
                model_bytes = numpy_to_model_bytes(model_record["model_bytes"])

                path = os.path.join(tmp_dir, f"model_r{server_round}_c{i}.json")
                with open(path, "wb") as f:
                    f.write(model_bytes)
                round_paths.append(path)

                # Log client metrics
                if "metrics" in reply.content:
                    m = reply.content["metrics"]
                    acc = m.get("train_acc", float("nan"))
                    auc = m.get("train_auc", float("nan"))
                    print(
                        f"  client {i}: acc={acc:.4f}  auc={auc:.4f}"
                        f"  examples={int(m.get('num_examples', 0))}"
                    )

            all_model_paths.extend(round_paths)
            print(
                f"[Server] Round {server_round} complete"
                f" — collected {len(round_paths)} models"
                f"  (total so far: {len(all_model_paths)})"
            )

        # ── Build FedXGBBagging ensemble ─────────────────────────────────
        print(
            f"\n[Server] Building FedXGBBagging ensemble"
            f" from {len(all_model_paths)} models …"
        )
        run_tag = f"federated_{num_rounds}rounds_{n_clients}clients"
        ensemble = FedXGBBagging(
            model_paths=all_model_paths,
            voting="soft",
            config={"bank_name_round_number": run_tag},
        )

        # ── Optional: distributed evaluation round ───────────────────────
        # Send the *first* collected model to ``fraction_evaluate`` of clients
        # as a representative model so they can report per-partition metrics.
        n_eval = max(1, int(fraction_evaluate * n_clients))
        eval_node_ids = node_ids[:n_eval]

        # Load the first model as the "representative" model for evaluation
        with open(all_model_paths[0], "rb") as f:
            rep_model_bytes = f.read()
        rep_model_array = np.frombuffer(rep_model_bytes, dtype=np.uint8).copy()

        eval_messages = [
            grid.create_message(
                content=RecordDict(
                    {"model": ArrayRecord({"model_bytes": rep_model_array})}
                ),
                message_type="evaluate",
                dst_node_id=nid,
                group_id="eval",
            )
            for nid in eval_node_ids
        ]
        eval_replies = list(grid.send_and_receive(eval_messages))

        eval_accs, eval_aucs = [], []
        for reply in eval_replies:
            if "metrics" in reply.content:
                m = reply.content["metrics"]
                eval_accs.append(m.get("eval_acc", 0.0))
                eval_aucs.append(m.get("eval_auc", 0.0))
        if eval_accs:
            print(
                f"\n[Server] Distributed eval"
                f" — avg_acc={np.mean(eval_accs):.4f}"
                f"  avg_auc={np.mean(eval_aucs):.4f}"
            )

        # ── Central (server-side) evaluation ────────────────────────────
        csv_path = _bundled_csv()
        if os.path.exists(csv_path):
            print("\n[Server] Central evaluation on held-out server data …")
            X_test, y_test = _load_central_test(csv_path)
            ensemble.test_data = X_test
            y_pred, y_prob = ensemble.predict(X_test)
            metrics = ensemble.evaluate_predictions(y_test, y_pred, y_prob)

            print("\n[Server] ══ Final Ensemble Metrics ══")
            for k, v in metrics.items():
                if v is not None:
                    print(f"  {k:12s}: {v:.4f}")

        # ── Persist final models ─────────────────────────────────────────
        output_dir = "final_ensemble"
        os.makedirs(output_dir, exist_ok=True)
        for path in all_model_paths:
            dst = os.path.join(output_dir, os.path.basename(path))
            shutil.copy2(path, dst)
        print(
            f"\n[Server] Saved {len(all_model_paths)} model files"
            f" to '{output_dir}/'"
        )
        print(
            f"[Server] To use the ensemble:\n"
            f"  from frauddetection.fed_xgb_bagging import FedXGBBagging\n"
            f"  ensemble = FedXGBBagging(\n"
            f"      model_paths=glob.glob('{output_dir}/*.json'),\n"
            f"      voting='soft',\n"
            f"      config={{'bank_name_round_number': '{run_tag}'}},\n"
            f"  )"
        )

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
