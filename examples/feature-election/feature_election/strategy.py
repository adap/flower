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
from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
)
from flwr.common import log
from flwr.common.record import Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy
from flwr.serverapp.strategy.fedavg import FedAvg
from flwr.serverapp.strategy.result import Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
)

logger = logging.getLogger(__name__)


class FeatureElectionStrategy(Strategy):
    """Feature Election Strategy that delegates FL rounds to FedAvg internals.

    Architecture:
    - Feature Election (Round 1): Custom logic
    - Tuning Rounds: Custom logic - evaluates freedom_degree candidates
    - FL Training Rounds: Delegates to FedAvg-style aggregation using Flower utilities
    """

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
        # FedAvg-compatible keys for FL rounds
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        metricrecord_key: str = "metrics",
    ):
        super().__init__()

        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ["weighted", "uniform"]:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")

        # Feature Election parameters
        self.freedom_degree = freedom_degree
        self.tuning_rounds = tuning_rounds
        self.aggregation_mode = aggregation_mode
        self.auto_tune = auto_tune
        self.skip_feature_election = skip_feature_election

        # Sampling parameters (shared with FedAvg pattern)
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.accept_failures = accept_failures

        # Keys for Flower's aggregate_* utilities
        self.weighted_by_key = weighted_by_key
        self.arrayrecord_key = arrayrecord_key
        self.metricrecord_key = metricrecord_key

        self.save_path = save_path

        # Feature election state
        self.global_feature_mask: Optional[np.ndarray] = None
        self.num_features: Optional[int] = None
        self.election_stats: Dict = {}
        self.cached_client_selections: Dict[str, Dict] = {}
        self.feature_names: Optional[List[str]] = None

        # Tuning state (Hill Climbing)
        self.tuning_history: List[Dict[str, float]] = []
        self.search_step = 0.1
        self.current_direction = 1

        # Communication tracking
        self.cumulative_communication_bytes: int = 0

    # ==========================================================================
    # Sampling - Simple, no unnecessary waiting
    # ==========================================================================

    def _sample_nodes(
        self, grid: Grid, fraction: float, min_nodes: int
    ) -> Tuple[List[int], int]:
        """Sample nodes without blocking waits.

        Unlike Flower's sample_nodes, this doesn't block waiting for nodes since we
        assume nodes are already connected when start() is called.
        """
        all_node_ids = list(grid.get_node_ids())
        num_available = len(all_node_ids)
        num_to_sample = max(int(num_available * fraction), min_nodes)

        if num_to_sample >= num_available:
            return all_node_ids, num_available
        return random.sample(all_node_ids, num_to_sample), num_available

    # ==========================================================================
    # Message Construction - FedAvg pattern
    # ==========================================================================

    def _construct_messages(
        self,
        arrays: ArrayRecord,
        config: ConfigRecord,
        node_ids: List[int],
        message_type: str,
    ) -> List[Message]:
        """Construct messages following FedAvg's pattern."""
        content = RecordDict(
            {
                self.arrayrecord_key: arrays,
                "config": config,
            }
        )
        return [
            Message(content=content, message_type=message_type, dst_node_id=node_id)
            for node_id in node_ids
        ]

    # ==========================================================================
    # Communication Tracking
    # ==========================================================================

    def _calculate_payload_size(self, arrays: ArrayRecord) -> int:
        """Calculate ArrayRecord size in bytes."""
        return sum(cast(Array, arrays[k]).numpy().nbytes for k in arrays.keys())

    def _track_communication(
        self, arrays: ArrayRecord, num_clients: int, direction: str, round_num: int
    ) -> None:
        """Track communication bytes."""
        size = self._calculate_payload_size(arrays) * num_clients
        self.cumulative_communication_bytes += size
        mb = size / (1024 * 1024)
        total_mb = self.cumulative_communication_bytes / (1024 * 1024)
        logger.info(
            f"Round {round_num} {direction}: {mb:.4f} MB (Total: {total_mb:.2f} MB)"
        )

    # ==========================================================================
    # Configure Methods
    # ==========================================================================

    def configure_train(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure training round."""
        node_ids, _ = self._sample_nodes(
            grid, self.fraction_train, self.min_train_nodes
        )

        train_config = ConfigRecord(dict(config))
        train_config["server_round"] = server_round

        # Determine phase and build payload
        is_election = server_round == 1 and not self.skip_feature_election
        is_tuning = (
            1 < server_round <= 1 + self.tuning_rounds
            and not self.skip_feature_election
        )

        payload = ArrayRecord()

        if is_election:
            train_config["phase"] = "feature_selection"
        elif is_tuning:
            train_config["phase"] = "tuning_eval"
            if not self.cached_client_selections:
                logger.error("No cached votes for tuning!")
                return []
            if self.global_feature_mask is not None:
                payload["feature_mask"] = Array(
                    self.global_feature_mask.astype(np.float32)
                )
            else:
                mask = self._aggregate_selections(self.cached_client_selections)
                payload["feature_mask"] = Array(mask.astype(np.float32))
        else:
            train_config["phase"] = "fl_training"
            for k, v in arrays.items():
                payload[k] = v

        self._track_communication(payload, len(node_ids), "downstream", server_round)
        return self._construct_messages(
            payload, train_config, node_ids, MessageType.TRAIN
        )

    def configure_evaluate(
        self,
        server_round: int,
        arrays: ArrayRecord,
        config: ConfigRecord,
        grid: Grid,
    ) -> Iterable[Message]:
        """Configure evaluation round."""
        if self.fraction_evaluate == 0.0:
            return []

        node_ids, _ = self._sample_nodes(
            grid, self.fraction_evaluate, self.min_evaluate_nodes
        )

        eval_config = ConfigRecord(dict(config))
        eval_config["server_round"] = server_round

        self._track_communication(arrays, len(node_ids), "downstream", server_round)
        return self._construct_messages(
            arrays, eval_config, node_ids, MessageType.EVALUATE
        )

    # ==========================================================================
    # Aggregation - Routes to appropriate handler
    # ==========================================================================

    def aggregate_train(
        self,
        server_round: int,
        results: Iterable[Message],
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Route aggregation based on phase."""
        valid_replies = [
            msg for msg in results if msg.has_content() and not msg.has_error()
        ]
        if not valid_replies:
            return None, None

        # Track upstream
        for msg in valid_replies:
            arrays = msg.content.get(self.arrayrecord_key, ArrayRecord())
            self.cumulative_communication_bytes += self._calculate_payload_size(arrays)

        is_election = server_round == 1 and not self.skip_feature_election
        is_tuning = (
            1 < server_round <= 1 + self.tuning_rounds
            and not self.skip_feature_election
        )

        if is_election:
            return self._aggregate_feature_election(valid_replies)
        elif is_tuning:
            return self._aggregate_tuning(valid_replies, server_round)
        else:
            return self._aggregate_fl_train(valid_replies)

    def _aggregate_fl_train(
        self, valid_replies: List[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate FL training using Flower's FedAvg."""
        records = []
        for msg in valid_replies:
            rd = RecordDict()
            rd[self.arrayrecord_key] = msg.content.get(
                self.arrayrecord_key, ArrayRecord()
            )
            rd[self.metricrecord_key] = msg.content.get(
                self.metricrecord_key, MetricRecord()
            )
            records.append(rd)

        if not records:
            return None, None

        agg_arrays = aggregate_arrayrecords(records, self.weighted_by_key)
        agg_metrics = aggregate_metricrecords(records, self.weighted_by_key)
        logger.info(
            "FL aggregation using Flower's aggregate_arrayrecords/metricrecords"
        )
        return agg_arrays, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: Iterable[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation using Flower's utilities."""
        valid_replies = [
            msg for msg in results if msg.has_content() and not msg.has_error()
        ]
        if not valid_replies:
            return None

        # Track upstream
        for msg in valid_replies:
            arrays = msg.content.get(self.arrayrecord_key, ArrayRecord())
            self.cumulative_communication_bytes += self._calculate_payload_size(arrays)

        # Use Flower's aggregate_metricrecords
        records = []
        for msg in valid_replies:
            rd = RecordDict()
            rd[self.metricrecord_key] = msg.content.get(
                self.metricrecord_key, MetricRecord()
            )
            records.append(rd)

        if not records:
            return None

        agg_metrics = aggregate_metricrecords(records, self.weighted_by_key)
        agg_metrics["total_bytes_transmitted"] = self.cumulative_communication_bytes
        return agg_metrics

    # ==========================================================================
    # Feature Election - Custom Logic
    # ==========================================================================

    def _aggregate_feature_election(
        self, valid_replies: List[Message]
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate feature election votes - custom logic."""
        logger.info("Round 1: Aggregating feature election votes")

        self.cached_client_selections = {}

        for msg in valid_replies:
            arrays = msg.content.get(self.arrayrecord_key, ArrayRecord())
            metrics = msg.content.get(self.metricrecord_key, MetricRecord())

            if "feature_mask" not in arrays or "feature_scores" not in arrays:
                continue

            mask = cast(Array, arrays["feature_mask"]).numpy().astype(bool)
            scores = cast(Array, arrays["feature_scores"]).numpy().astype(float)
            num_samples = metrics.get(self.weighted_by_key, 0)
            num_samples = (
                int(num_samples) if isinstance(num_samples, (int, float)) else 0
            )

            if self.num_features is None:
                self.num_features = len(mask)
                if self.feature_names is None:
                    self.feature_names = [
                        f"feature_{i:03d}" for i in range(self.num_features)
                    ]

            self.cached_client_selections[str(msg.metadata.src_node_id)] = {
                "selected_features": mask,
                "feature_scores": scores,
                "num_samples": num_samples,
            }

        if not self.cached_client_selections:
            logger.warning("No valid feature selections received")
            return None, None

        # Compute the global mask with the INITIAL freedom_degree
        # This mask will be evaluated in the first tuning round
        self.global_feature_mask = self._aggregate_selections(
            self.cached_client_selections
        )
        self._calculate_statistics()

        # The first tuning round (Round 2) will:
        #   1. Evaluate the current freedom_degree with clients
        #   2. Record the actual score in tuning_history
        #   3. Then compute the next freedom_degree

        logger.info(
            f"Feature election complete. Initial FD={self.freedom_degree:.4f}, "
            f"selected {int(np.sum(self.global_feature_mask))} features. "
            f"Will be evaluated in first tuning round."
        )

        return (
            ArrayRecord(
                {"feature_mask": Array(self.global_feature_mask.astype(np.float32))}
            ),
            MetricRecord(
                {
                    "num_features_selected": int(np.sum(self.global_feature_mask)),
                    "num_features_original": int(len(self.global_feature_mask)),
                    "freedom_degree": self.freedom_degree,
                    "total_bytes_transmitted": self.cumulative_communication_bytes,
                }
            ),
        )

    def _aggregate_tuning(
        self, valid_replies: List[Message], server_round: int
    ) -> Tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate tuning evaluation results."""
        scores = []
        for msg in valid_replies:
            metrics = msg.content.get(self.metricrecord_key, MetricRecord())
            val_acc = metrics.get("val_accuracy", 0.0)
            if isinstance(val_acc, (int, float)):
                scores.append(float(val_acc))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        logger.info(f"Tuning: FD={self.freedom_degree:.4f} -> score={avg_score:.4f}")

        self.tuning_history.append(
            {
                "freedom_degree": self.freedom_degree,
                "score": avg_score,
                "num_features_selected": (
                    int(np.sum(self.global_feature_mask))
                    if self.global_feature_mask is not None
                    else 0
                ),
            }
        )

        # Determine if we should compute next freedom_degree or finalize
        is_last_tuning_round = server_round >= 1 + self.tuning_rounds

        if not is_last_tuning_round:
            # Compute next freedom_degree for the next tuning round
            # first_step=True only when this is the very first entry in history
            is_first_step = len(self.tuning_history) == 1
            self.freedom_degree = self._next_freedom_degree(first_step=is_first_step)

            # Recompute global mask with the NEW freedom_degree
            self.global_feature_mask = self._aggregate_selections(
                self.cached_client_selections
            )
            self._calculate_statistics()

            logger.info(
                f"Next tuning round will evaluate FD={self.freedom_degree:.4f} "
                f"({int(np.sum(self.global_feature_mask))} features)"
            )
        else:
            # Last tuning round - select the best freedom_degree
            best_entry = max(self.tuning_history, key=lambda x: x["score"])
            best_fd = best_entry["freedom_degree"]
            best_score = best_entry["score"]
            logger.info(
                f"Tuning complete. Best: FD={best_fd:.4f} (score={best_score:.4f})"
            )
            # Set to best and recompute final mask
            self.freedom_degree = best_fd
            self.global_feature_mask = self._aggregate_selections(
                self.cached_client_selections
            )
            self._calculate_statistics()

        agg_arrays = ArrayRecord()
        if self.global_feature_mask is not None:
            agg_arrays["feature_mask"] = Array(
                self.global_feature_mask.astype(np.float32)
            )

        return agg_arrays, MetricRecord(
            {
                "val_accuracy": avg_score,
                "freedom_degree": self.freedom_degree,
                "total_bytes_transmitted": self.cumulative_communication_bytes,
            }
        )

    def _aggregate_selections(self, client_selections: Dict[str, Dict]) -> np.ndarray:
        """Aggregate client feature selections using freedom_degree."""
        masks = np.array([s["selected_features"] for s in client_selections.values()])
        scores = np.array([s["feature_scores"] for s in client_selections.values()])
        samples = np.array([s["num_samples"] for s in client_selections.values()])

        total = samples.sum()
        weights = samples / total if total > 0 else np.ones(len(samples)) / len(samples)

        intersection = np.all(masks, axis=0)
        union = np.any(masks, axis=0)

        if self.freedom_degree == 0:
            return intersection
        if self.freedom_degree == 1:
            return union

        # Weighted election for difference set
        difference = union & ~intersection
        if not np.any(difference):
            return intersection

        # Normalize and weight scores
        scaled = np.zeros_like(scores, dtype=float)
        for i, (mask, score) in enumerate(zip(masks, scores)):
            selected = mask.astype(bool)
            if np.any(selected):
                sel_scores = score[selected]
                min_s, max_s = sel_scores.min(), sel_scores.max()
                rng = max_s - min_s
                if rng > 0:
                    scaled[i][selected] = (sel_scores - min_s) / rng
                else:
                    scaled[i][selected] = 1.0
            scaled[i][intersection] = 0.0
            if self.aggregation_mode == "weighted":
                scaled[i] *= weights[i]

        aggregated = scaled.sum(axis=0)
        n_additional = int(np.ceil(difference.sum() * self.freedom_degree))

        if n_additional > 0:
            diff_idx = np.where(difference)[0]
            diff_scores = aggregated[difference]
            k = min(n_additional, len(diff_scores))
            top_idx = np.argpartition(diff_scores, -k)[-k:]

            selected_diff = np.zeros_like(difference)
            selected_diff[diff_idx[top_idx]] = True
            return intersection | selected_diff

        return intersection

    def _calculate_statistics(self) -> None:
        """Calculate election statistics."""
        if not self.cached_client_selections or self.global_feature_mask is None:
            return

        masks = np.array(
            [s["selected_features"] for s in self.cached_client_selections.values()]
        )

        self.election_stats = {
            "num_clients": len(self.cached_client_selections),
            "num_features_original": int(self.num_features or 0),
            "num_features_selected": int(np.sum(self.global_feature_mask)),
            "reduction_ratio": 1.0
            - (np.sum(self.global_feature_mask) / (self.num_features or 1)),
            "freedom_degree": self.freedom_degree,
            "intersection_features": int(np.all(masks, axis=0).sum()),
            "union_features": int(np.any(masks, axis=0).sum()),
        }

    def _next_freedom_degree(self, first_step: bool = False) -> float:
        """Hill climbing for freedom_degree."""
        MIN_FD, MAX_FD = 0.05, 1.0

        if first_step:
            return float(
                np.clip(self.freedom_degree + self.search_step, MIN_FD, MAX_FD)
            )

        if len(self.tuning_history) < 2:
            return self.freedom_degree

        curr_fd = self.tuning_history[-1]["freedom_degree"]
        curr_score = self.tuning_history[-1]["score"]
        prev_fd = self.tuning_history[-2]["freedom_degree"]
        prev_score = self.tuning_history[-2]["score"]

        if curr_score > prev_score:
            new_fd = curr_fd + self.current_direction * self.search_step
        else:
            self.current_direction *= -1
            self.search_step *= 0.5
            new_fd = prev_fd + self.current_direction * self.search_step

        return float(np.clip(new_fd, MIN_FD, MAX_FD))

    # ==========================================================================
    # Public Interface
    # ==========================================================================

    def summary(self) -> None:
        """Log strategy configuration."""
        log(logging.INFO, "\tFeature Election Settings:")
        log(logging.INFO, "\t\tfreedom_degree: %.2f", self.freedom_degree)
        log(logging.INFO, "\t\ttuning_rounds: %d", self.tuning_rounds)
        log(logging.INFO, "\t\taggregation_mode: %s", self.aggregation_mode)
        log(logging.INFO, "\tSampling:")
        log(logging.INFO, "\t\tfraction_train: %.2f", self.fraction_train)
        log(logging.INFO, "\t\tfraction_evaluate: %.2f", self.fraction_evaluate)

    def get_results(self) -> Dict:
        """Get results dictionary."""
        feature_selection_details = None
        if self.global_feature_mask is not None and self.feature_names is not None:
            feature_selection_details = {
                name: bool(selected)
                for name, selected in zip(self.feature_names, self.global_feature_mask)
            }

        return {
            "global_feature_mask": (
                self.global_feature_mask.tolist()
                if self.global_feature_mask is not None
                else None
            ),
            "feature_selection_with_names": feature_selection_details,
            "selected_feature_names": (
                [
                    name
                    for name, sel in zip(self.feature_names, self.global_feature_mask)
                    if sel
                ]
                if self.global_feature_mask is not None
                and self.feature_names is not None
                else None
            ),
            "election_stats": self.election_stats,
            "tuning_history": self.tuning_history,
            "total_bytes_transmitted": self.cumulative_communication_bytes,
        }

    def save_results(self, filename: str = "feature_election_results.json") -> None:
        """Save results to JSON."""
        path = self.save_path / filename if self.save_path else Path(filename)
        with open(path, "w") as f:
            json.dump(self.get_results(), f, indent=2)
        logger.info(f"Results saved to {path}")
        self._save_client_selections()

    def _save_client_selections(
        self, filename: str = "client_feature_selections.json"
    ) -> None:
        """Save per-client feature selections to a separate file."""
        if not self.cached_client_selections or self.feature_names is None:
            return

        path = self.save_path / filename if self.save_path else Path(filename)

        client_details = {}
        for client_id, selection_data in self.cached_client_selections.items():
            mask = selection_data["selected_features"]
            scores = selection_data["feature_scores"]
            num_samples = selection_data["num_samples"]

            features_info = {}
            for i, (name, selected, score) in enumerate(
                zip(self.feature_names, mask, scores)
            ):
                features_info[name] = {
                    "selected": bool(selected),
                    "score": float(score),
                }

            selected_names = [
                name for name, sel in zip(self.feature_names, mask) if sel
            ]

            client_details[f"client_{client_id}"] = {
                "num_samples": int(num_samples),
                "num_features_selected": int(np.sum(mask)),
                "selected_feature_names": selected_names,
                "all_features": features_info,
            }

        if len(self.cached_client_selections) > 1:
            masks = np.array(
                [s["selected_features"] for s in self.cached_client_selections.values()]
            )
            intersection_mask = np.all(masks, axis=0)
            union_mask = np.any(masks, axis=0)

            client_details["_summary"] = {
                "total_clients": len(self.cached_client_selections),
                "features_selected_by_all": [
                    name
                    for name, sel in zip(self.feature_names, intersection_mask)
                    if sel
                ],
                "features_selected_by_any": [
                    name for name, sel in zip(self.feature_names, union_mask) if sel
                ],
                "num_intersection": int(np.sum(intersection_mask)),
                "num_union": int(np.sum(union_mask)),
            }

        with open(path, "w") as f:
            json.dump(client_details, f, indent=2)
        logger.info(f"Client selections saved to {path}")

    # ==========================================================================
    # Main Workflow
    # ==========================================================================

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
        """Execute Feature Election + FL workflow."""
        t_start = time.time()

        log(logging.INFO, "Starting %s", self.__class__.__name__)
        self.summary()

        train_config = train_config or ConfigRecord()
        evaluate_config = evaluate_config or ConfigRecord()

        # Setup output directory
        run_dir = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        self.save_path = Path.cwd() / f"outputs/{run_dir}"
        self.save_path.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print("Feature Election + Federated Learning")
        print(
            f"  Rounds: {num_rounds} | FD: {self.freedom_degree} | Tuning: {self.tuning_rounds}"
        )
        print(f"  Output: {self.save_path}")
        print("=" * 60)

        result = Result()
        global_mask: Optional[np.ndarray] = None
        global_weights: Optional[ArrayRecord] = None

        setup_rounds = (1 + self.tuning_rounds) if not self.skip_feature_election else 0

        for rnd in range(1, num_rounds + 1):
            is_setup = rnd <= setup_rounds
            phase = "ELECTION/TUNING" if is_setup else "FL TRAINING"
            log(logging.INFO, f"[Round {rnd}/{num_rounds}] {phase}")

            # Prepare input arrays
            input_arrays = ArrayRecord()
            if not is_setup:
                if global_mask is not None:
                    input_arrays["feature_mask"] = Array(global_mask.astype(np.float32))
                if global_weights:
                    for k, v in global_weights.items():
                        input_arrays[k] = v

            # Train
            messages = self.configure_train(
                rnd, input_arrays, ConfigRecord(dict(train_config)), grid
            )
            replies = list(grid.send_and_receive(messages=messages, timeout=timeout))

            agg_arrays, agg_metrics = self.aggregate_train(rnd, replies)

            if agg_arrays and "feature_mask" in agg_arrays:
                global_mask = (
                    cast(Array, agg_arrays["feature_mask"]).numpy().astype(bool)
                )
                self.global_feature_mask = global_mask

            if not is_setup and agg_arrays:
                global_weights = agg_arrays
                result.arrays = agg_arrays

            if agg_metrics:
                result.train_metrics_clientapp[rnd] = agg_metrics

            # Save after setup
            if rnd == setup_rounds and setup_rounds > 0:
                self.save_results()

            # Evaluation (FL phase only)
            if not is_setup and self.fraction_evaluate > 0 and global_weights:
                eval_arrays = ArrayRecord()
                if global_mask is not None:
                    eval_arrays["feature_mask"] = Array(global_mask.astype(np.float32))
                for k, v in global_weights.items():
                    eval_arrays[k] = v

                eval_msgs = self.configure_evaluate(
                    rnd, eval_arrays, ConfigRecord(dict(evaluate_config)), grid
                )
                if eval_msgs:
                    eval_replies = grid.send_and_receive(
                        messages=eval_msgs, timeout=timeout
                    )
                    eval_metrics = self.aggregate_evaluate(rnd, eval_replies)
                    if eval_metrics:
                        result.evaluate_metrics_clientapp[rnd] = eval_metrics

            if evaluate_fn and global_weights:
                res = evaluate_fn(rnd, global_weights)
                if res:
                    result.evaluate_metrics_serverapp[rnd] = res

        print("=" * 60)
        final_mb = self.cumulative_communication_bytes / (1024 * 1024)
        log(
            logging.INFO,
            f"Complete! Communication: {final_mb:.2f} MB | Time: {time.time()-t_start:.1f}s",
        )

        return result
