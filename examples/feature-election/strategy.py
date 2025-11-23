"""
Feature Election Strategy for Flower

Implements the Feature Election algorithm for federated feature selection.
Aggregates client feature selections using weighted voting based on freedom_degree.

Key parameters:
- freedom_degree: Controls feature selection (0=intersection, 1=union)
- aggregation_mode: 'weighted' or 'uniform' aggregation
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

logger = logging.getLogger(__name__)


class FeatureElectionStrategy(Strategy):
    """
    Feature Election Strategy for Flower.

    Implements federated feature selection with configurable aggregation:
    - Client-side: Each client performs local feature selection
    - Server-side: Aggregates selections using weighted voting

    The freedom_degree parameter controls the balance between:
    - 0.0: Intersection (only features selected by ALL clients)
    - 1.0: Union (features selected by ANY client)
    - 0.0-1.0: Weighted selection from difference set
    """

    def __init__(
        self,
        freedom_degree: float = 0.5,
        aggregation_mode: str = "weighted",
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn=None,
        on_fit_config_fn=None,
        on_evaluate_config_fn=None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
    ):
        """
        Initialize Feature Election Strategy.

        Args:
            freedom_degree: Feature selection parameter (0=intersection, 1=union)
            aggregation_mode: 'weighted' or 'uniform' aggregation
            fraction_fit: Fraction of clients to sample for training
            fraction_evaluate: Fraction of clients to sample for evaluation
            min_fit_clients: Minimum clients for training
            min_evaluate_clients: Minimum clients for evaluation
            min_available_clients: Minimum clients that must be available
            evaluate_fn: Optional server-side evaluation function
            on_fit_config_fn: Function to configure fit
            on_evaluate_config_fn: Function to configure evaluation
            accept_failures: Whether to accept client failures
            initial_parameters: Initial model parameters (optional)
        """
        super().__init__()

        # Validate parameters
        if not 0 <= freedom_degree <= 1:
            raise ValueError("freedom_degree must be between 0 and 1")
        if aggregation_mode not in ["weighted", "uniform"]:
            raise ValueError("aggregation_mode must be 'weighted' or 'uniform'")

        self.freedom_degree = freedom_degree
        self.aggregation_mode = aggregation_mode
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters

        # Results storage
        self.global_feature_mask: Optional[np.ndarray] = None
        self.client_scores: Dict[str, Dict] = {}
        self.num_features: Optional[int] = None
        self.election_stats: Dict = {}

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training (feature selection)."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create fit config
        config = {"server_round": server_round}
        if self.on_fit_config_fn is not None:
            config = self.on_fit_config_fn(server_round)

        # Create fit instructions
        fit_ins = FitIns(parameters, config)

        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate feature selections from clients.

        Implements the core Feature Election algorithm:
        1. Collect client feature selections and scores
        2. Calculate intersection and union
        3. Apply weighted voting based on freedom_degree
        """

        if not results:
            logger.warning("No results received from clients")
            return None, {}

        if not self.accept_failures and failures:
            logger.warning(f"Received {len(failures)} failures, not accepting")
            return None, {}

        # Extract client selections from parameters
        client_selections = self._extract_client_selections(results)

        if not client_selections:
            logger.warning("No valid client selections received")
            return None, {}

        # Run feature election algorithm
        self.global_feature_mask = self._aggregate_selections(client_selections)

        # Calculate statistics
        self._calculate_statistics(client_selections)

        # Package results as parameters
        # Encode the global feature mask as a float32 array
        global_mask_array = self.global_feature_mask.astype(np.float32)
        aggregated_parameters = ndarrays_to_parameters([global_mask_array])

        logger.info(
            f"Feature election completed: {np.sum(self.global_feature_mask)}/"
            f"{len(self.global_feature_mask)} features selected"
        )

        return aggregated_parameters, self.election_stats

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""

        if self.fraction_evaluate == 0.0:
            return []

        # Sample clients
        sample_size, min_num_clients = self.num_evaluate_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create evaluate config
        config = {}
        if self.on_evaluate_config_fn is not None:
            config = self.on_evaluate_config_fn(server_round)

        # Create evaluate instructions
        evaluate_ins = EvaluateIns(parameters, config)

        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""

        if not results:
            return None, {}

        # Weighted average of metrics
        total_loss = 0.0
        total_accuracy = 0.0
        total_samples = 0

        for _, eval_res in results:
            total_loss += eval_res.loss * eval_res.num_examples
            total_samples += eval_res.num_examples
            if "accuracy" in eval_res.metrics:
                total_accuracy += eval_res.metrics["accuracy"] * eval_res.num_examples

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_accuracy = total_accuracy / total_samples if total_samples > 0 else 0.0

        return avg_loss, {"accuracy": avg_accuracy}

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model parameters (server-side)."""

        if self.evaluate_fn is None:
            return None

        # Extract feature mask from parameters
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        if len(parameters_ndarrays) > 0:
            feature_mask = parameters_ndarrays[0].astype(bool)
            return self.evaluate_fn(server_round, feature_mask)

        return None

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluate_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def _extract_client_selections(
        self, results: List[Tuple[ClientProxy, FitRes]]
    ) -> Dict[str, Dict]:
        """
        Extract and validate client selection data from parameters.

        Expected parameter format from clients:
        - Array 0: Binary feature mask (float32)
        - Array 1: Feature importance scores (float32)
        """
        client_selections = {}

        for client, fit_res in results:
            try:
                # Extract arrays from parameters
                arrays = parameters_to_ndarrays(fit_res.parameters)

                if len(arrays) < 2:
                    logger.warning(
                        f"Client {client.cid} returned insufficient arrays: "
                        f"expected 2, got {len(arrays)}"
                    )
                    continue

                # Extract mask and scores
                selected_features = arrays[0].astype(bool)
                feature_scores = arrays[1].astype(float)

                # Get metrics
                metrics = fit_res.metrics
                num_samples = fit_res.num_examples

                # Validate
                if len(selected_features) == 0 or len(feature_scores) == 0:
                    logger.warning(f"Client {client.cid} returned empty selection")
                    continue

                if len(selected_features) != len(feature_scores):
                    logger.warning(
                        f"Client {client.cid} selection length mismatch: "
                        f"{len(selected_features)} vs {len(feature_scores)}"
                    )
                    continue

                # Set num_features on first valid response
                if self.num_features is None:
                    self.num_features = len(selected_features)
                elif len(selected_features) != self.num_features:
                    logger.warning(
                        f"Client {client.cid} has wrong number of features: "
                        f"{len(selected_features)} vs {self.num_features}"
                    )
                    continue

                client_selections[client.cid] = {
                    "selected_features": selected_features,
                    "feature_scores": feature_scores,
                    "num_samples": num_samples,
                    "initial_score": float(metrics.get("initial_score", 0.0)),
                    "fs_score": float(metrics.get("fs_score", 0.0)),
                }

                logger.info(
                    f"Client {client.cid}: {np.sum(selected_features)} features, "
                    f"score {metrics.get('initial_score', 0):.4f} -> "
                    f"{metrics.get('fs_score', 0):.4f}"
                )

            except Exception as e:
                logger.error(f"Error processing client {client.cid}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return client_selections

    def _aggregate_selections(self, client_selections: Dict[str, Dict]) -> np.ndarray:
        """
        Core Feature Election algorithm implementation.
        
        Aggregates client selections using weighted voting based on freedom_degree.
        """
        
        num_clients = len(client_selections)
        logger.info(f"Aggregating selections from {num_clients} clients")

        # Convert to numpy arrays
        masks = []
        scores = []
        weights = []
        total_samples = 0

        for client_name, selection in client_selections.items():
            masks.append(selection["selected_features"])
            scores.append(selection["feature_scores"])
            num_samples = selection["num_samples"]
            weights.append(num_samples)
            total_samples += num_samples

            # Store client scores
            self.client_scores[client_name] = {
                "initial_score": selection["initial_score"],
                "fs_score": selection["fs_score"],
                "num_features": int(np.sum(selection["selected_features"])),
                "num_samples": num_samples,
            }

        masks = np.array(masks)
        scores = np.array(scores)
        weights = np.array(weights) / total_samples if total_samples > 0 else np.ones(len(weights)) / len(weights)

        # Calculate intersection and union
        intersection_mask = self._get_intersection(masks)
        union_mask = self._get_union(masks)

        logger.info(f"Intersection: {np.sum(intersection_mask)} features")
        logger.info(f"Union: {np.sum(union_mask)} features")

        # Handle edge cases
        if self.freedom_degree == 0:
            global_mask = intersection_mask
        elif self.freedom_degree == 1:
            global_mask = union_mask
        else:
            # Main algorithm: weighted election
            global_mask = self._weighted_election(
                masks, scores, weights, intersection_mask, union_mask
            )

        logger.info(f"Global mask: {np.sum(global_mask)} features selected")

        return global_mask

    def _weighted_election(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        weights: np.ndarray,
        intersection_mask: np.ndarray,
        union_mask: np.ndarray,
    ) -> np.ndarray:
        """Perform weighted election for features in (union - intersection)."""
        
        # Get difference set
        difference_mask = union_mask & ~intersection_mask

        if not np.any(difference_mask):
            return intersection_mask

        # Scale scores and apply weights
        scaled_scores = np.zeros_like(scores)

        for i, (client_mask, client_scores) in enumerate(zip(masks, scores)):
            selected = client_mask.astype(bool)

            if np.any(selected):
                selected_scores = client_scores[selected]
                if len(selected_scores) > 0:
                    min_score = np.min(selected_scores)
                    max_score = np.max(selected_scores)
                    range_score = max_score - min_score

                    if range_score > 0:
                        scaled_scores[i][selected] = (client_scores[selected] - min_score) / range_score
                    else:
                        scaled_scores[i][selected] = 1.0

            # Zero out intersection features
            scaled_scores[i][intersection_mask] = 0.0

            # Apply client weight if in weighted mode
            if self.aggregation_mode == "weighted":
                scaled_scores[i] *= weights[i]

        # Aggregate scores across clients
        aggregated_scores = np.sum(scaled_scores, axis=0)

        # Select top features from difference set
        n_additional = int(np.ceil(np.sum(difference_mask) * self.freedom_degree))

        if n_additional > 0:
            diff_indices = np.where(difference_mask)[0]
            diff_scores = aggregated_scores[difference_mask]

            if len(diff_scores) > 0:
                k = -min(n_additional, len(diff_scores))
                top_indices = np.argpartition(diff_scores, k)[k:]

                selected_difference = np.zeros_like(difference_mask)
                selected_difference[diff_indices[top_indices]] = True

                global_mask = intersection_mask | selected_difference
            else:
                global_mask = intersection_mask
        else:
            global_mask = intersection_mask

        return global_mask

    @staticmethod
    def _get_intersection(masks: np.ndarray) -> np.ndarray:
        """Get intersection of all feature masks."""
        return np.all(masks, axis=0)

    @staticmethod
    def _get_union(masks: np.ndarray) -> np.ndarray:
        """Get union of all feature masks."""
        return np.any(masks, axis=0)

    def _calculate_statistics(self, client_selections: Dict[str, Dict]) -> None:
        """Calculate and store election statistics."""
        
        masks = np.array([sel["selected_features"] for sel in client_selections.values()])
        intersection_mask = self._get_intersection(masks)
        union_mask = self._get_union(masks)

        self.election_stats = {
            "num_clients": len(client_selections),
            "num_features_original": int(self.num_features) if self.num_features else 0,
            "num_features_selected": int(np.sum(self.global_feature_mask)) if self.global_feature_mask is not None else 0,
            "reduction_ratio": float(
                1 - (np.sum(self.global_feature_mask) / len(self.global_feature_mask))
                if self.global_feature_mask is not None else 0
            ),
            "freedom_degree": float(self.freedom_degree),
            "aggregation_mode": self.aggregation_mode,
            "intersection_features": int(np.sum(intersection_mask)),
            "union_features": int(np.sum(union_mask)),
        }

    def get_results(self) -> Dict:
        """Get feature election results."""
        return {
            "global_feature_mask": self.global_feature_mask.tolist() if self.global_feature_mask is not None else None,
            "election_stats": self.election_stats,
            "client_scores": self.client_scores,
        }
