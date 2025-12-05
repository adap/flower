"""
Feature Election Server for Flower

Implements a multi-phase workflow:
1. Feature Election (Round 1): Collect feature votes.
2. Tuning (Rounds 2 to 1+Tuning): Iteratively tune freedom_degree using Hill Climbing.
3. Federated Learning (Remaining Rounds): Train model using the best selected features.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, cast

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.common import log
from flwr.common.record import Array
from flwr.serverapp import Grid, ServerApp

from .strategy import FeatureElectionStrategy

logger = logging.getLogger(__name__)

# Create ServerApp
app = ServerApp()


def create_run_dir(config: dict) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        serializable_config = {k: v for k, v in config.items()}
        json.dump(serializable_config, fp, indent=2, default=str)

    return save_path, run_dir


def aggregate_model_weights(results: Iterable[Message]) -> Optional[ArrayRecord]:
    """Aggregate model weights using FedAvg."""
    results_list = list(results)
    valid_results = [msg for msg in results_list if msg.has_content()]

    if not valid_results:
        return None

    # Collect weights and sample counts
    all_weights = []
    total_samples = 0

    for msg in valid_results:
        arrays = msg.content.get("arrays", ArrayRecord())
        metrics = msg.content.get("metrics", MetricRecord())

        if "model_weights" not in arrays:
            continue

        # Extract weights (coefficients + intercept)
        weights_list = []
        i = 0
        while f"weight_{i}" in arrays:
            w_arr = cast(Array, arrays[f"weight_{i}"])
            weights_list.append(w_arr.numpy())
            i += 1

        if len(weights_list) < 2:
            continue

        num_ex = metrics.get("num-examples", 1)
        if isinstance(num_ex, (int, float)):
            num_samples = int(num_ex)
        else:
            num_samples = 1

        all_weights.append((weights_list, num_samples))
        total_samples += num_samples

    if not all_weights or total_samples == 0:
        return None

    # Weighted average
    # We assume all clients have the same model architecture
    num_weight_arrays = len(all_weights[0][0])
    aggregated_weights = []

    for i in range(num_weight_arrays):
        weighted_sum = np.zeros_like(all_weights[0][0][i])
        for weights_list, num_samples in all_weights:
            weighted_sum += weights_list[i] * (num_samples / total_samples)
        aggregated_weights.append(weighted_sum)

    # Package result
    result = ArrayRecord()
    for i, w in enumerate(aggregated_weights):
        result[f"weight_{i}"] = Array(w.astype(np.float32))
    # Add flag to indicate weights are present
    result["model_weights"] = Array(np.array([1.0], dtype=np.float32))

    return result


def aggregate_fl_metrics(results: Iterable[Message]) -> Optional[MetricRecord]:
    """Aggregate FL training metrics."""
    results_list = list(results)
    valid_results = [msg for msg in results_list if msg.has_content()]

    if not valid_results:
        return None

    total_train_acc = 0.0
    total_val_acc = 0.0
    total_loss = 0.0
    total_samples = 0

    for msg in valid_results:
        metrics = msg.content.get("metrics", MetricRecord())
        num_ex = metrics.get("num-examples", 0)

        if isinstance(num_ex, (int, float)):
            num_samples = int(num_ex)
        else:
            num_samples = 0

        if num_samples > 0:
            train_acc_raw = metrics.get("train_accuracy", 0)
            val_acc_raw = metrics.get("val_accuracy", 0)
            train_loss_raw = metrics.get("train_loss", 0)

            train_acc = float(train_acc_raw) if isinstance(train_acc_raw, (int, float)) else 0.0
            val_acc = float(val_acc_raw) if isinstance(val_acc_raw, (int, float)) else 0.0
            train_loss = float(train_loss_raw) if isinstance(train_loss_raw, (int, float)) else 0.0

            total_train_acc += train_acc * num_samples
            total_val_acc += val_acc * num_samples
            total_loss += train_loss * num_samples
            total_samples += num_samples

    if total_samples == 0:
        return None

    return MetricRecord(
        {
            "train_accuracy": total_train_acc / total_samples,
            "val_accuracy": total_val_acc / total_samples,
            "train_loss": total_loss / total_samples,
            "num-examples": total_samples,
        }
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """
    Main entry point for the Feature Election + FL ServerApp.
    """
    # =======================================================
    # Read Configuration
    # =======================================================
    run_config = context.run_config

    skip_feature_election = bool(run_config.get("skip-feature-election", False))
    auto_tune = bool(run_config.get("auto-tune", False))
    tuning_rounds = int(str(run_config.get("tuning-rounds", 0)))
    freedom_degree = float(run_config.get("freedom-degree", 0.5))
    aggregation_mode = str(run_config.get("aggregation-mode", "weighted"))

    num_rounds = int(str(run_config.get("num-rounds", 5)))
    fraction_train = float(run_config.get("fraction-train", 1.0))
    fraction_evaluate = float(run_config.get("fraction-evaluate", 1.0))
    min_train_nodes = int(str(run_config.get("min-train-nodes", 2)))
    min_evaluate_nodes = int(str(run_config.get("min-evaluate-nodes", 2)))
    fs_method = str(run_config.get("fs-method", "lasso"))

    save_path, run_dir = create_run_dir(dict(run_config))

    print("=" * 70)
    print("Feature Election + Federated Learning")
    print("=" * 70)
    print(f"  Feature selection method: {fs_method}")
    print(f"  Total rounds: {num_rounds}")
    print(f"  Skip feature election: {skip_feature_election}")
    print(f"  Auto-tune: {auto_tune}")
    if auto_tune and not skip_feature_election:
        print(f"  Tuning rounds: {tuning_rounds}")
    print(f"  Output directory: {save_path}")
    print("=" * 70)

    # Initialize Strategy
    strategy = FeatureElectionStrategy(
        freedom_degree=freedom_degree,
        tuning_rounds=tuning_rounds if (auto_tune and not skip_feature_election) else 0,
        aggregation_mode=aggregation_mode,
        auto_tune=auto_tune,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=min_train_nodes,
        min_evaluate_nodes=min_evaluate_nodes,
        accept_failures=True,
        save_path=save_path,
    )

    global_feature_mask = None
    global_model_weights = None

    results_history: Dict[str, Dict] = {
        "feature_selection": {},
        "tuning": {},
        "fl_training": {},
        "evaluation": {},
    }

    total_setup_rounds = 1 + strategy.tuning_rounds if not skip_feature_election else 0

    log(logging.INFO, "Starting Feature Election + FL workflow")

    # =======================================================
    # Main Loop
    # =======================================================
    for current_round in range(1, num_rounds + 1):
        log(logging.INFO, "")

        is_setup_phase = current_round <= total_setup_rounds
        input_arrays = ArrayRecord()

        if is_setup_phase:
            log(
                logging.INFO,
                f"[ROUND {current_round}/{num_rounds}] - FEATURE ELECTION / TUNING PHASE",
            )
        else:
            log(logging.INFO, f"[ROUND {current_round}/{num_rounds}] - FL TRAINING PHASE")

            if not skip_feature_election and global_feature_mask is None:
                log(logging.ERROR, "Cannot proceed to FL: Feature election failed (no mask).")
                break

            if global_feature_mask is not None:
                input_arrays["feature_mask"] = Array(global_feature_mask.astype(np.float32))

            if global_model_weights is not None:
                for key in global_model_weights:
                    input_arrays[key] = global_model_weights[key]

        # Configure Train
        train_config = ConfigRecord()
        train_messages = strategy.configure_train(current_round, input_arrays, train_config, grid)

        train_replies = grid.send_and_receive(
            messages=train_messages,
            timeout=3600,
        )

        # Aggregation
        if is_setup_phase:
            agg_arrays, agg_metrics = strategy.aggregate_train(current_round, train_replies)

            if agg_arrays is not None and "feature_mask" in agg_arrays:
                # FIX: Explicit cast
                mask_arr = cast(Array, agg_arrays["feature_mask"])
                global_feature_mask = mask_arr.numpy().astype(bool)
                num_sel = np.sum(global_feature_mask)
                log(logging.INFO, f"  Current global feature mask: {num_sel} features")

            if agg_metrics is not None:
                metrics_dict = dict(agg_metrics)

                # Format bytes to MB
                total_bytes = metrics_dict.get("total_bytes_transmitted", 0)
                if isinstance(total_bytes, (int, float)):
                    metrics_dict["total_mb_transmitted"] = float(f"{total_bytes / (1024*1024):.2f}")

                if current_round == 1:
                    log(logging.INFO, f"  Election Metrics: {metrics_dict}")
                    results_history["feature_selection"][current_round] = metrics_dict
                else:
                    log(logging.INFO, f"  Tuning Metrics: {metrics_dict}")
                    results_history["tuning"][current_round] = metrics_dict

            if current_round == total_setup_rounds:
                log(logging.INFO, "  Setup phase complete. Finalizing feature election results.")
                strategy.save_results("feature_election_results.json")

        else:
            # IMPORTANT: Call strategy.aggregate_train() to update communication counters!
            # We ignore the return value because we do FedAvg aggregation manually below.
            strategy.aggregate_train(current_round, train_replies)

            # FL Aggregation (FedAvg)
            current_weights = aggregate_model_weights(train_replies)

            if current_weights is not None:
                global_model_weights = current_weights
                log(logging.INFO, "  Global model weights updated via FedAvg")
            else:
                log(logging.WARNING, "  Could not aggregate weights this round (insufficient data)")

            fl_metrics = aggregate_fl_metrics(train_replies)
            if fl_metrics is not None:
                metrics_dict = dict(fl_metrics)

                # --- Inject Communication Metrics ---
                total_bytes = strategy.cumulative_communication_bytes
                metrics_dict["total_mb_transmitted"] = float(f"{total_bytes / (1024*1024):.2f}")

                log(logging.INFO, f"  FL Training metrics: {metrics_dict}")
                results_history["fl_training"][current_round] = metrics_dict

            # Evaluation
            if fraction_evaluate > 0 and global_model_weights is not None:
                eval_arrays = ArrayRecord()
                if global_feature_mask is not None:
                    eval_arrays["feature_mask"] = Array(global_feature_mask.astype(np.float32))
                for key in global_model_weights:
                    eval_arrays[key] = global_model_weights[key]

                eval_config = ConfigRecord({"server_round": current_round})

                eval_messages = strategy.configure_evaluate(
                    current_round, eval_arrays, eval_config, grid
                )

                if eval_messages:
                    eval_replies = grid.send_and_receive(
                        messages=eval_messages,
                        timeout=3600,
                    )
                    eval_metrics = strategy.aggregate_evaluate(current_round, eval_replies)

                    if eval_metrics is not None:
                        metrics_dict = dict(eval_metrics)

                        # Format bytes
                        total_bytes = metrics_dict.get("total_bytes_transmitted", 0)
                        if isinstance(total_bytes, (int, float)):
                            metrics_dict["total_mb_transmitted"] = float(
                                f"{total_bytes / (1024*1024):.2f}"
                            )

                        log(logging.INFO, f"  Evaluation metrics: {metrics_dict}")
                        results_history["evaluation"][current_round] = metrics_dict

    # =======================================================
    # Final Summary
    # =======================================================
    log(logging.INFO, "")
    log(logging.INFO, "=" * 50)
    log(logging.INFO, "Workflow completed!")

    final_bytes = strategy.cumulative_communication_bytes
    final_mb = final_bytes / (1024 * 1024)
    log(logging.INFO, f"Total Communication Cost: {final_mb:.2f} MB ({final_bytes} bytes)")

    log(logging.INFO, "=" * 50)

    # Save results
    with open(save_path / "results_history.json", "w") as f:
        json.dump(results_history, f, indent=2)

    if global_model_weights is not None:
        weights_to_save = {}
        for key in global_model_weights:
            if key.startswith("weight_"):
                w_arr = cast(Array, global_model_weights[key])
                weights_to_save[key] = w_arr.numpy().tolist()
        with open(save_path / "final_model_weights.json", "w") as f:
            json.dump(weights_to_save, f, indent=2)

    log(logging.INFO, f"Results saved to: {save_path}")
