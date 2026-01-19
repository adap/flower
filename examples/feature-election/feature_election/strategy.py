"""Feature Election Strategy for Flower.

Implements the Feature Election algorithm for federated feature selection. Aggregates
client feature selection decisions using weighted voting based on freedom_degree.
Supports iterative auto-tuning via Hill Climbing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, cast
import random

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict
from flwr.common.record import Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy

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

    def _calculate_payload_size(self, arrays: ArrayRecord) -> int:
        """Helper to calculate size in bytes of an ArrayRecord."""
        total_bytes = 0
        for key in arrays.keys():
            arr = cast(Array, arrays[key]).numpy()
            total_bytes += arr.nbytes
        return total_bytes

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
