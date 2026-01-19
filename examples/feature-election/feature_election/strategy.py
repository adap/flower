"""feature-election: Federated feature selection with Flower.

Implements the Feature Election algorithm for federated feature selection.
Aggregates client feature selection decisions using weighted voting based on
freedom_degree. Supports iterative auto-tuning via Hill Climbing.

The strategy orchestrates a multi-phase workflow:
1. Feature Election (Round 1): Collect feature votes from clients.
2. Tuning (Rounds 2 to 1+tuning_rounds): Iteratively tune freedom_degree.
3. Federated Learning (Remaining Rounds): Train model using selected features.
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common import log
from flwr.common.record import Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.result import Result

MEGABYTE_SIZE = 1024
logger = logging.getLogger(__name__)


class FeatureElectionStrategy(Strategy):
    """Feature Election Strategy for Flower using Message API."""

    def __init__(
        self,
        freedom_degree: float = 0.5,
        tuning_rounds: int = 0,
        aggregation_mode: str = "weighted",
        auto_tune: bool = False,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        accept_failures: bool = True,
        save_path: Optional[Path] = None,
        skip_feature_election: bool = False,
    ):
        super().__init__()

        # Validate parameters
        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ["weighted", "uniform"]:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")

        self.freedom_degree = freedom_degree
        self.tuning_rounds = tuning_rounds
        self.aggregation_mode = aggregation_mode
        self.auto_tune = auto_tune
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.accept_failures = accept_failures
        self.save_path = save_path
        self.skip_feature_election = skip_feature_election

        # Results storage
        self.global_feature_mask: Optional[np.ndarray] = None
        self.client_scores: Dict[str, Dict] = {}
        self.num_features: Optional[int] = None
        self.election_stats: Dict = {}

        # State for caching Round 1 results
        self.cached_client_selections: Dict[str, Dict] = {}

        # State for Tuning Search (Hill Climbing)
        self.tuning_history: List[Tuple[float, float]] = []  # [(fd, score)]
        self.search_step = 0.1
        self.current_direction = 1  # 1 for increasing FD, -1 for decreasing

        # Metrics: Communication Cost Tracking
        self.cumulative_communication_bytes: int = 0

    def get_summary(self) -> Dict[str, Any]:
        """Return a summary of the strategy configuration."""
        return {
            "freedom_degree": self.freedom_degree,
            "tuning_rounds": self.tuning_rounds,
            "aggregation_mode": self.aggregation_mode,
            "auto_tune": self.auto_tune,
            "fraction_train": self.fraction_train,
            "fraction_evaluate": self.fraction_evaluate,
            "total_bytes_transmitted": self.cumulative_communication_bytes,
        }

    def summary(self) -> None:
        """Required by the base Strategy class."""
        # We log the info so it's still useful, but return None to satisfy Mypy/Flower
        logger.info(f"Strategy Configuration: {self.get_summary()}")

    def set_save_path(self, path: Path) -> None:
        """Set the path where results will be saved."""
        self.save_path = path

    def _create_run_dir(self, config: dict) -> Tuple[Path, str]:
        """Create a directory where to save results from this run."""
        current_time = datetime.now()
        run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
        save_path = Path.cwd() / f"outputs/{run_dir}"
        save_path.mkdir(parents=True, exist_ok=True)

        with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
            serializable_config = {k: v for k, v in config.items()}
            json.dump(serializable_config, fp, indent=2, default=str)

        return save_path, run_dir

    def _calculate_payload_size(self, arrays: ArrayRecord) -> int:
        """Helper to calculate size in bytes of an ArrayRecord."""
        total_bytes = 0
        for key in arrays.keys():
            arr = cast(Array, arrays[key]).numpy()
            total_bytes += arr.nbytes
        return total_bytes

    def _aggregate_model_weights(
        self, results: Iterable[Message]
    ) -> Optional[ArrayRecord]:
        """Aggregate model weights using FedAvg."""
        results_list = list(results)
        valid_results = [msg for msg in results_list if msg.has_content()]

        if not valid_results:
            return None

        # Collect weights and sample counts
        all_weights: List[Tuple[List[np.ndarray], int]] = []
        total_samples = 0

        for msg in valid_results:
            arrays = msg.content.get("arrays", ArrayRecord())
            metrics = msg.content.get("metrics", MetricRecord())

            if "model_weights" not in arrays:
                continue

            # Extract weights (coefficients + intercept)
            weights_list: List[np.ndarray] = []
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
        aggregated_weights: List[np.ndarray] = []

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

    def _aggregate_fl_metrics(
        self, results: Iterable[Message]
    ) -> Optional[MetricRecord]:
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

                train_acc = (
                    float(train_acc_raw)
                    if isinstance(train_acc_raw, (int, float))
                    else 0.0
                )
                val_acc = (
                    float(val_acc_raw) if isinstance(val_acc_raw, (int, float)) else 0.0
                )
                train_loss = (
                    float(train_loss_raw)
                    if isinstance(train_loss_raw, (int, float))
                    else 0.0
                )

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

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        """Execute the Feature Election + Federated Learning workflow.

        This method orchestrates a multi-phase workflow:
        1. Feature Election (Round 1): Collect feature votes from clients
        2. Tuning (Rounds 2 to 1+tuning_rounds): Iteratively tune freedom_degree
        3. Federated Learning (Remaining Rounds): Train model using selected features

        Parameters
        ----------
        grid : Grid
            The Grid instance used for node sampling and communication.
        initial_arrays : ArrayRecord
            Initial model parameters (unused for feature election, but required by interface).
        num_rounds : int (default: 3)
            Total number of rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation (not used in this workflow).

        Returns
        -------
        Result
            Results containing training and evaluation metrics from all rounds.
        """
        t_start = time.time()

        # Log startup info
        log(logging.INFO, "Starting %s strategy:", self.__class__.__name__)
        self.summary()
        log(logging.INFO, "")

        # Initialize configs if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config

        # Create run directory and set save path
        run_config_dict = {
            "num_rounds": num_rounds,
            "freedom_degree": self.freedom_degree,
            "tuning_rounds": self.tuning_rounds,
            "aggregation_mode": self.aggregation_mode,
            "auto_tune": self.auto_tune,
            "skip_feature_election": self.skip_feature_election,
            "fraction_train": self.fraction_train,
            "fraction_evaluate": self.fraction_evaluate,
        }
        save_path, run_dir = self._create_run_dir(run_config_dict)
        self.save_path = save_path

        # Print configuration banner
        print("=" * 70)
        print("Feature Election + Federated Learning")
        print("=" * 70)
        print(f"  Total rounds: {num_rounds}")
        print(f"  Skip feature election: {self.skip_feature_election}")
        print(f"  Auto-tune: {self.auto_tune}")
        if self.auto_tune and not self.skip_feature_election:
            print(f"  Tuning rounds: {self.tuning_rounds}")
        print(f"  Output directory: {save_path}")
        print("=" * 70)

        # Initialize result object
        result = Result()

        # State variables for the workflow
        global_feature_mask: Optional[np.ndarray] = None
        global_model_weights: Optional[ArrayRecord] = None

        results_history: Dict[str, Dict] = {
            "feature_selection": {},
            "tuning": {},
            "fl_training": {},
            "evaluation": {},
        }

        # Calculate setup rounds (feature election + tuning)
        total_setup_rounds = (
            1 + self.tuning_rounds if not self.skip_feature_election else 0
        )

        log(logging.INFO, "Starting Feature Election + FL workflow")

        # Evaluate initial parameters if evaluate_fn provided
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(logging.INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

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
                log(
                    logging.INFO,
                    f"[ROUND {current_round}/{num_rounds}] - FL TRAINING PHASE",
                )

                if not self.skip_feature_election and global_feature_mask is None:
                    log(
                        logging.ERROR,
                        "Cannot proceed to FL: Feature election failed (no mask).",
                    )
                    break

                if global_feature_mask is not None:
                    input_arrays["feature_mask"] = Array(
                        global_feature_mask.astype(np.float32)
                    )

                if global_model_weights is not None:
                    for key in global_model_weights:
                        input_arrays[key] = global_model_weights[key]

            # Configure Train
            train_messages = self.configure_train(
                current_round, input_arrays, ConfigRecord(dict(train_config)), grid
            )

            train_replies = list(
                grid.send_and_receive(
                    messages=train_messages,
                    timeout=timeout,
                )
            )

            # Aggregation
            if is_setup_phase:
                agg_arrays, agg_metrics = self.aggregate_train(
                    current_round, train_replies
                )

                if agg_arrays is not None and "feature_mask" in agg_arrays:
                    mask_arr = cast(Array, agg_arrays["feature_mask"])
                    global_feature_mask = mask_arr.numpy().astype(bool)
                    # Also update instance variable
                    self.global_feature_mask = global_feature_mask
                    num_sel = int(np.sum(global_feature_mask))
                    log(
                        logging.INFO,
                        f"  Current global feature mask: {num_sel} features",
                    )

                if agg_metrics is not None:
                    metrics_dict = dict(agg_metrics)

                    # Format bytes to MB
                    total_bytes = metrics_dict.get("total_bytes_transmitted", 0)
                    if isinstance(total_bytes, (int, float)):
                        metrics_dict["total_mb_transmitted"] = float(
                            f"{total_bytes / (1024*1024):.2f}"
                        )

                    if current_round == 1:
                        log(logging.INFO, f"  Election Metrics: {metrics_dict}")
                        results_history["feature_selection"][
                            current_round
                        ] = metrics_dict
                        result.train_metrics_clientapp[current_round] = agg_metrics
                    else:
                        log(logging.INFO, f"  Tuning Metrics: {metrics_dict}")
                        results_history["tuning"][current_round] = metrics_dict
                        result.train_metrics_clientapp[current_round] = agg_metrics

                if current_round == total_setup_rounds:
                    log(
                        logging.INFO,
                        "  Setup phase complete. Finalizing feature election results.",
                    )
                    self.save_results("feature_election_results.json")

            else:
                # FL Phase
                # Call aggregate_train to update communication counters
                self.aggregate_train(current_round, train_replies)

                # FL Aggregation (FedAvg)
                current_weights = self._aggregate_model_weights(train_replies)

                if current_weights is not None:
                    global_model_weights = current_weights
                    result.arrays = current_weights
                    log(logging.INFO, "  Global model weights updated via FedAvg")
                else:
                    log(
                        logging.WARNING,
                        "  Could not aggregate weights this round (insufficient data)",
                    )

                fl_metrics = self._aggregate_fl_metrics(train_replies)
                if fl_metrics is not None:
                    metrics_dict = dict(fl_metrics)

                    # Inject Communication Metrics
                    total_bytes = self.cumulative_communication_bytes
                    metrics_dict["total_mb_transmitted"] = float(
                        f"{total_bytes / (1024*1024):.2f}"
                    )

                    log(logging.INFO, f"  FL Training metrics: {metrics_dict}")
                    results_history["fl_training"][current_round] = metrics_dict
                    result.train_metrics_clientapp[current_round] = fl_metrics

                # Evaluation
                if self.fraction_evaluate > 0 and global_model_weights is not None:
                    eval_arrays = ArrayRecord()
                    if global_feature_mask is not None:
                        eval_arrays["feature_mask"] = Array(
                            global_feature_mask.astype(np.float32)
                        )
                    for key in global_model_weights:
                        eval_arrays[key] = global_model_weights[key]

                    eval_config_round = ConfigRecord(dict(evaluate_config))
                    eval_config_round["server_round"] = current_round

                    eval_messages = self.configure_evaluate(
                        current_round, eval_arrays, eval_config_round, grid
                    )

                    if eval_messages:
                        eval_replies = grid.send_and_receive(
                            messages=eval_messages,
                            timeout=timeout,
                        )
                        eval_metrics = self.aggregate_evaluate(
                            current_round, eval_replies
                        )

                        if eval_metrics is not None:
                            metrics_dict = dict(eval_metrics)

                            # Format bytes
                            total_bytes_eval = metrics_dict.get(
                                "total_bytes_transmitted", 0
                            )
                            if isinstance(total_bytes_eval, (int, float)):
                                metrics_dict["total_mb_transmitted"] = float(
                                    f"{total_bytes_eval / (1024*1024):.2f}"
                                )

                            log(logging.INFO, f"  Evaluation metrics: {metrics_dict}")
                            results_history["evaluation"][current_round] = metrics_dict
                            result.evaluate_metrics_clientapp[current_round] = (
                                eval_metrics
                            )

                # Centralized evaluation if provided
                if evaluate_fn and global_model_weights is not None:
                    log(logging.INFO, "Global evaluation")
                    res = evaluate_fn(current_round, global_model_weights)
                    log(logging.INFO, "\t└──> MetricRecord: %s", res)
                    if res is not None:
                        result.evaluate_metrics_serverapp[current_round] = res

        # =======================================================
        # Final Summary
        # =======================================================
        log(logging.INFO, "")
        log(logging.INFO, "=" * 50)
        log(logging.INFO, "Workflow completed!")

        final_bytes = self.cumulative_communication_bytes
        final_mb = final_bytes / (1024 * 1024)
        log(
            logging.INFO,
            f"Total Communication Cost: {final_mb:.2f} MB ({final_bytes} bytes)",
        )

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

        log(logging.INFO, "")
        log(logging.INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(logging.INFO, "")

        return result

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure the next round of training."""

        # --- Logic to determine Phase ---
        is_collection = server_round == 1
        is_tuning = 1 < server_round <= 1 + self.tuning_rounds

        # Determine number of nodes to sample
        all_node_ids = list(grid.get_node_ids())
        num_available = len(all_node_ids)
        num_nodes = max(int(num_available * self.fraction_train), self.min_train_nodes)

        if num_nodes < num_available:
            node_ids = random.sample(all_node_ids, num_nodes)
        else:
            node_ids = all_node_ids

        # Base Configuration
        train_config = ConfigRecord(dict(config))
        train_config["server_round"] = server_round

        # Prepare Payload Arrays
        payload_arrays = ArrayRecord()

        if is_collection:
            train_config["phase"] = "feature_selection"

        elif is_tuning:
            train_config["phase"] = "tuning_eval"

            if not self.cached_client_selections:
                logger.error(
                    "No cached votes found for tuning phase! Did Round 1 fail?"
                )
                return []

            # Generate mask on the fly
            current_mask = self._aggregate_selections(self.cached_client_selections)
            payload_arrays["feature_mask"] = Array(current_mask.astype(np.float32))
            logger.info(
                f"Tuning Round {server_round}: Testing freedom_degree={self.freedom_degree:.4f}"
            )

        else:
            # Standard FL Phase
            train_config["phase"] = "fl_training"
            for k, v in arrays.items():
                payload_arrays[k] = v

        # --- METRIC: Track Downstream Bytes ---
        payload_size = self._calculate_payload_size(payload_arrays)
        total_downstream = payload_size * len(node_ids)
        self.cumulative_communication_bytes += total_downstream

        # Human readable logging
        mb_sent = total_downstream / (MEGABYTE_SIZE * MEGABYTE_SIZE)
        logger.info(
            f"Round {server_round} Downstream: {mb_sent:.4f} MB ({len(node_ids)} clients)"
        )

        # Construct Messages
        content = RecordDict({"arrays": payload_arrays, "config": train_config})
        messages = []
        for node_id in node_ids:
            message = Message(
                content=content,
                message_type="train",
                dst_node_id=node_id,
            )
            messages.append(message)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        results: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate results from clients."""
        if not results:
            return None, None

        # Convert Iterable to List for multiple iterations
        results_list = list(results)
        valid_results = [msg for msg in results_list if msg.has_content()]
        if not valid_results:
            return None, None

        # --- METRIC: Track Upstream Bytes ---
        round_upstream_bytes = 0
        for msg in valid_results:
            arrays = cast(ArrayRecord, msg.content.get("arrays", ArrayRecord()))
            round_upstream_bytes += self._calculate_payload_size(arrays)

        self.cumulative_communication_bytes += round_upstream_bytes
        mb_recv = round_upstream_bytes / (MEGABYTE_SIZE * MEGABYTE_SIZE)
        total_mb = self.cumulative_communication_bytes / (MEGABYTE_SIZE * MEGABYTE_SIZE)
        logger.info(
            f"Round {server_round} Upstream: {mb_recv:.4f} MB. Total Session: {total_mb:.2f} MB"
        )

        is_collection = server_round == 1
        is_tuning = 1 < server_round <= 1 + self.tuning_rounds

        if is_collection:
            # --- PHASE 1: COLLECT VOTES ---
            logger.info("Aggregating Round 1: Extracting Feature Votes")
            self.cached_client_selections = self._extract_client_selections(
                valid_results
            )

            if not self.cached_client_selections:
                logger.warning("No valid selections found in Round 1")
                return None, None

            self.global_feature_mask = self._aggregate_selections(
                self.cached_client_selections
            )
            self._calculate_statistics(self.cached_client_selections)

            self.tuning_history.append((self.freedom_degree, 0.0))

            if self.tuning_rounds > 0:
                self.freedom_degree = self._calculate_next_fd(first_step=True)

            agg_arrays = ArrayRecord(
                {
                    "feature_mask": Array(
                        cast(np.ndarray, self.global_feature_mask).astype(np.float32)
                    )
                }
            )

            metrics = MetricRecord(
                {
                    "num_features_selected": int(np.sum(self.global_feature_mask)),
                    "num_features_original": int(len(self.global_feature_mask)),
                    "freedom_degree": float(self.freedom_degree),
                    "total_bytes_transmitted": self.cumulative_communication_bytes,
                }
            )
            return agg_arrays, metrics

        elif is_tuning:
            # --- PHASE 2: TUNING EVALUATION ---
            total_score = 0.0
            count = 0
            for msg in valid_results:
                metrics = cast(MetricRecord, msg.content.get("metrics", MetricRecord()))
                val_acc_raw = metrics.get("val_accuracy", 0.0)

                if isinstance(val_acc_raw, (float, int)):
                    val_acc = float(val_acc_raw)
                else:
                    val_acc = 0.0

                total_score += val_acc
                count += 1

            avg_score = total_score / count if count > 0 else 0.0

            logger.info(
                f"Tuning Result: FD={self.freedom_degree:.4f} -> Score={avg_score:.4f}"
            )
            self.tuning_history.append((self.freedom_degree, avg_score))

            if server_round < 1 + self.tuning_rounds:
                self.freedom_degree = self._calculate_next_fd(first_step=False)
            else:
                best_fd, best_score = max(self.tuning_history, key=lambda x: x[1])
                logger.info(
                    f"Tuning Complete. Winner: FD={best_fd:.4f} (Score={best_score:.4f})"
                )

                self.freedom_degree = best_fd
                self.global_feature_mask = self._aggregate_selections(
                    self.cached_client_selections
                )
                self._calculate_statistics(self.cached_client_selections)

            agg_arrays = ArrayRecord()
            if self.global_feature_mask is not None:
                agg_arrays["feature_mask"] = Array(
                    self.global_feature_mask.astype(np.float32)
                )
            metrics = MetricRecord(
                {
                    "val_accuracy": avg_score,
                    "freedom_degree": self.freedom_degree,
                    "total_bytes_transmitted": self.cumulative_communication_bytes,
                }
            )
            return agg_arrays, metrics

        else:
            # --- PHASE 3: FL TRAINING ---
            return None, None

    def _calculate_next_fd(self, first_step: bool = False) -> float:
        """Hill Climbing Logic with Step Decay."""
        MIN_FD = 0.05
        MAX_FD = 1.0

        if first_step:
            new_fd = self.freedom_degree + self.search_step
            return float(np.clip(new_fd, MIN_FD, MAX_FD))

        if len(self.tuning_history) < 2:
            return self.freedom_degree

        curr_fd, curr_score = self.tuning_history[-1]
        prev_fd, prev_score = self.tuning_history[-2]

        if curr_score > prev_score:
            logger.info("Score improved! Continuing direction.")
            new_fd = curr_fd + (self.current_direction * self.search_step)
        else:
            logger.info(
                f"Score dropped (curr={curr_score:.4f} < prev={prev_score:.4f}). Reversing and refining."
            )
            # Reverse Direction
            self.current_direction *= -1
            # Decay Step Size
            self.search_step *= 0.5
            new_fd = prev_fd + (self.current_direction * self.search_step)

        return float(np.clip(new_fd, MIN_FD, MAX_FD))

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure evaluation."""
        if self.fraction_evaluate == 0.0:
            return []

        all_node_ids = list(grid.get_node_ids())
        num_nodes = max(
            int(len(all_node_ids) * self.fraction_evaluate), self.min_evaluate_nodes
        )

        if num_nodes < len(all_node_ids):
            node_ids = random.sample(all_node_ids, num_nodes)
        else:
            node_ids = all_node_ids

        eval_config = ConfigRecord(dict(config))
        eval_config["server_round"] = server_round

        # --- METRIC: Track Downstream Bytes (Evaluation) ---
        payload_size = self._calculate_payload_size(arrays)
        total_downstream = payload_size * len(node_ids)
        self.cumulative_communication_bytes += total_downstream

        content = RecordDict({"arrays": arrays, "config": eval_config})

        messages = []
        for node_id in node_ids:
            message = Message(
                content=content,
                message_type="evaluate",
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def aggregate_evaluate(
        self,
        server_round: int,
        results: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation metrics."""
        if not results:
            return None

        results_list = list(results)
        valid_results = [msg for msg in results_list if msg.has_content()]
        if not valid_results:
            return None

        # --- METRIC: Track Upstream Bytes (Evaluation) ---
        round_upstream_bytes = 0
        for msg in valid_results:
            arrays = cast(ArrayRecord, msg.content.get("arrays", ArrayRecord()))
            round_upstream_bytes += self._calculate_payload_size(arrays)
        self.cumulative_communication_bytes += round_upstream_bytes

        # Log total cost
        total_mb = self.cumulative_communication_bytes / (MEGABYTE_SIZE * MEGABYTE_SIZE)
        logger.info(
            f"Total Communication Cost (Round {server_round}): {total_mb:.2f} MB"
        )

        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for msg in valid_results:
            metrics = cast(MetricRecord, msg.content.get("metrics", MetricRecord()))
            num_ex = metrics.get("num-examples", 0)

            if isinstance(num_ex, (int, float)):
                num_examples = int(num_ex)
            else:
                num_examples = 0

            if num_examples > 0:
                eval_loss_raw = metrics.get("eval_loss", 0)
                eval_accuracy_raw = metrics.get("eval_accuracy", 0)

                eval_loss = (
                    float(eval_loss_raw)
                    if isinstance(eval_loss_raw, (int, float))
                    else 0.0
                )
                eval_accuracy = (
                    float(eval_accuracy_raw)
                    if isinstance(eval_accuracy_raw, (int, float))
                    else 0.0
                )

                total_loss += eval_loss * num_examples
                total_accuracy += eval_accuracy * num_examples
                total_samples += num_examples

        if total_samples == 0:
            return None

        return MetricRecord(
            {
                "eval_loss": total_loss / total_samples,
                "eval_accuracy": total_accuracy / total_samples,
                "num-examples": total_samples,
                "total_bytes_transmitted": self.cumulative_communication_bytes,
            }
        )

    def _extract_client_selections(self, results: List[Message]) -> Dict[str, Dict]:
        """Extract client selection data from messages."""
        client_selections = {}

        for msg in results:
            try:
                content = msg.content
                arrays = content.get("arrays", ArrayRecord())
                metrics = content.get("metrics", MetricRecord())

                if "feature_mask" not in arrays or "feature_scores" not in arrays:
                    continue

                mask_arr = cast(Array, arrays["feature_mask"])
                score_arr = cast(Array, arrays["feature_scores"])

                selected_features = mask_arr.numpy().astype(bool)
                feature_scores = score_arr.numpy().astype(float)

                num_ex = metrics.get("num-examples", 0)
                if isinstance(num_ex, (int, float)):
                    num_samples = int(num_ex)
                else:
                    num_samples = 0

                # Set num_features if not set
                if self.num_features is None:
                    self.num_features = len(selected_features)

                init_score_raw = metrics.get("initial_score", 0.0)
                fs_score_raw = metrics.get("fs_score", 0.0)

                init_score = (
                    float(init_score_raw)
                    if isinstance(init_score_raw, (int, float))
                    else 0.0
                )
                fs_score = (
                    float(fs_score_raw)
                    if isinstance(fs_score_raw, (int, float))
                    else 0.0
                )

                client_selections[str(msg.metadata.src_node_id)] = {
                    "selected_features": selected_features,
                    "feature_scores": feature_scores,
                    "num_samples": num_samples,
                    "initial_score": init_score,
                    "fs_score": fs_score,
                }
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                continue

        return client_selections

    def _aggregate_selections(self, client_selections: Dict[str, Dict]) -> np.ndarray:
        """Aggregate client selections based on current self.freedom_degree."""
        masks = []
        scores = []
        weights_list = []
        total_samples = 0

        for client_name, selection in client_selections.items():
            masks.append(selection["selected_features"])
            scores.append(selection["feature_scores"])
            num_samples = selection["num_samples"]
            weights_list.append(num_samples)
            total_samples += num_samples

        masks_np = np.array(masks)
        scores_np = np.array(scores)
        # Avoid division by zero
        weights = (
            np.array(weights_list) / total_samples
            if total_samples > 0
            else np.ones(len(weights_list)) / len(weights_list)
        )

        intersection_mask = self._get_intersection(masks_np)
        union_mask = self._get_union(masks_np)

        if self.freedom_degree == 0:
            global_mask = intersection_mask
        elif self.freedom_degree == 1:
            global_mask = union_mask
        else:
            global_mask = self._weighted_election(
                masks_np, scores_np, weights, intersection_mask, union_mask
            )

        return global_mask

    def _weighted_election(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        intersection_mask: np.ndarray,
        union_mask: np.ndarray,
    ) -> np.ndarray:
        """Perform weighted election."""
        difference_mask = union_mask & ~intersection_mask

        if not np.any(difference_mask):
            return intersection_mask

        # Calculate weighted scores for difference set
        scaled_scores = np.zeros_like(scores)

        for i, (client_mask, client_scores) in enumerate(zip(masks, scores)):
            selected = client_mask.astype(bool)

            if np.any(selected):
                # Normalize client scores 0-1
                sel_scores = client_scores[selected]
                if len(sel_scores) > 0:
                    min_s = np.min(sel_scores)
                    rng = np.max(sel_scores) - min_s
                    if rng > 0:
                        scaled_scores[i][selected] = (sel_scores - min_s) / rng
                    else:
                        scaled_scores[i][selected] = 1.0

            # Zero out intersection (already selected)
            scaled_scores[i][intersection_mask] = 0.0

            if self.aggregation_mode == "weighted":
                scaled_scores[i] *= weights[i]

        aggregated_scores = np.sum(scaled_scores, axis=0)

        # Determine how many additional features to pick
        n_additional = int(np.ceil(np.sum(difference_mask) * self.freedom_degree))

        if n_additional > 0:
            diff_indices = np.where(difference_mask)[0]
            diff_scores = aggregated_scores[difference_mask]

            if len(diff_scores) > 0:
                # Top-k selection
                k = -min(n_additional, len(diff_scores))
                top_indices: np.ndarray = np.argpartition(diff_scores, k)[k:]

                selected_difference = np.zeros_like(difference_mask)
                selected_difference[diff_indices[top_indices]] = True

                global_mask: np.ndarray = intersection_mask | selected_difference
                return global_mask

        return intersection_mask

    @staticmethod
    def _get_intersection(masks: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.all(masks, axis=0))

    @staticmethod
    def _get_union(masks: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.any(masks, axis=0))

    def _calculate_statistics(self, client_selections: Dict[str, Dict]) -> None:
        masks = np.array(
            [sel["selected_features"] for sel in client_selections.values()]
        )
        intersection_mask = self._get_intersection(masks)
        union_mask = self._get_union(masks)

        num_sel = (
            int(np.sum(self.global_feature_mask))
            if self.global_feature_mask is not None
            else 0
        )
        total = int(self.num_features) if self.num_features else 1

        self.election_stats = {
            "num_clients": len(client_selections),
            "num_features_original": total,
            "num_features_selected": num_sel,
            "reduction_ratio": 1.0 - (num_sel / total),
            "freedom_degree": float(self.freedom_degree),
            "intersection_features": int(np.sum(intersection_mask)),
            "union_features": int(np.sum(union_mask)),
        }

    def get_results(self) -> Dict:
        """Get results dictionary."""
        return {
            "global_feature_mask": (
                self.global_feature_mask.tolist()
                if self.global_feature_mask is not None
                else None
            ),
            "election_stats": self.election_stats,
            "tuning_history": self.tuning_history,
            "total_bytes_transmitted": self.cumulative_communication_bytes,
        }

    def save_results(self, filename: str = "feature_election_results.json") -> None:
        """Save results to JSON file."""
        if self.save_path is None:
            output_path = Path(filename)
        else:
            output_path = self.save_path / filename

        try:
            results = self.get_results()
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
