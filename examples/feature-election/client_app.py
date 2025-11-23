"""
Feature Election Client for Flower

Implements client-side feature selection and communicates with the server.
Supports multiple feature selection methods and evaluation metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    Context,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from feature_election_utils import FeatureSelector
from task import load_client_data

logger = logging.getLogger(__name__)


class FeatureElectionClient(Client):
    """
    Feature Election Client for Flower.

    Performs local feature selection and responds to server requests.
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1",
    ):
        """
        Initialize Feature Election Client.

        Args:
            client_id: Unique client identifier
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, uses X_train if None)
            y_val: Validation labels (optional, uses y_train if None)
            feature_names: Feature names (optional)
            fs_method: Feature selection method
            fs_params: Parameters for feature selection method
            eval_metric: Evaluation metric ('f1', 'accuracy', 'auc')
        """
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val if X_val is not None else X_train
        self.y_val = y_val if y_val is not None else y_train

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Initialize feature selector
        self.selector = FeatureSelector(
            fs_method=fs_method,
            fs_params=fs_params or {},
            eval_metric=eval_metric,
        )

        # Results storage
        self.selected_features: Optional[np.ndarray] = None
        self.feature_scores: Optional[np.ndarray] = None
        self.global_feature_mask: Optional[np.ndarray] = None

        logger.info(
            f"Client {client_id} initialized: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        Return current feature selection state as parameters.

        For Feature Election, we encode the selected feature mask as parameters.
        If no selection has been made yet, returns an empty mask.
        """
        if self.selected_features is not None:
            # Return the current feature selection mask
            mask_array = self.selected_features.astype(np.float32)
            parameters = ndarrays_to_parameters([mask_array])
        else:
            # No selection yet - return empty mask of correct size
            empty_mask = np.zeros(self.X_train.shape[1], dtype=np.float32)
            parameters = ndarrays_to_parameters([empty_mask])

        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        """
        Perform local feature selection.

        Returns:
            FitRes with feature selection parameters and metrics.
            Parameters contain the binary feature mask as a float32 array.
            Metrics contain feature scores and performance statistics.
        """
        server_round = ins.config.get("server_round", 0)
        logger.info(f"Client {self.client_id} starting feature selection (round {server_round})")

        try:
            # Perform feature selection
            selected_mask, feature_scores = self.selector.select_features(
                self.X_train, self.y_train
            )

            # Evaluate performance
            initial_score = self.selector.evaluate_model(
                self.X_train, self.y_train, self.X_val, self.y_val
            )

            # Apply feature mask and evaluate
            X_train_selected = self.X_train[:, selected_mask]
            X_val_selected = self.X_val[:, selected_mask]
            fs_score = self.selector.evaluate_model(
                X_train_selected, self.y_train, X_val_selected, self.y_val
            )

            # Store results
            self.selected_features = selected_mask
            self.feature_scores = feature_scores

            n_selected = np.sum(selected_mask)
            n_total = len(selected_mask)

            logger.info(
                f"Client {self.client_id}: Selected {n_selected}/{n_total} features, "
                f"score: {initial_score:.4f} -> {fs_score:.4f}"
            )

            # Package parameters: encode feature mask and scores as arrays
            # - Array 0: Binary feature mask (float32)
            # - Array 1: Feature importance scores (float32)
            mask_array = selected_mask.astype(np.float32)
            scores_array = feature_scores.astype(np.float32)
            parameters = ndarrays_to_parameters([mask_array, scores_array])

            # Return scalar metrics only (Flower constraint)
            metrics = {
                "initial_score": float(initial_score),
                "fs_score": float(fs_score),
                "num_selected": int(n_selected),
                "num_total": int(n_total),
                "client_id": int(self.client_id),
            }

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=parameters,
                num_examples=len(self.X_train),
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Client {self.client_id} feature selection failed: {e}")
            import traceback
            traceback.print_exc()

            # Return error status with empty selection
            n_features = self.X_train.shape[1]
            empty_mask = np.zeros(n_features, dtype=np.float32)
            empty_scores = np.zeros(n_features, dtype=np.float32)
            parameters = ndarrays_to_parameters([empty_mask, empty_scores])

            return FitRes(
                status=Status(code=Code.OK, message=f"Error: {str(e)}"),
                parameters=parameters,
                num_examples=len(self.X_train),
                metrics={
                    "initial_score": 0.0,
                    "fs_score": 0.0,
                    "num_selected": 0,
                    "num_total": int(n_features),
                    "client_id": int(self.client_id),
                    "error": str(e),
                },
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Evaluate model with selected features.

        Expected parameters format:
        - Array 0: Global feature mask (float32, binary values)

        Returns:
            EvaluateRes with loss and accuracy metrics
        """
        try:
            # Check if we have a global mask from server
            if len(ins.parameters.tensors) > 0:
                arrays = parameters_to_ndarrays(ins.parameters)

                if len(arrays) > 0 and len(arrays[0]) > 0:
                    global_mask = arrays[0].astype(bool)

                    # Validate mask size
                    if len(global_mask) != self.X_train.shape[1]:
                        logger.warning(
                            f"Client {self.client_id}: Global mask size mismatch: "
                            f"received {len(global_mask)}, expected {self.X_train.shape[1]}. "
                            f"Using local selection instead."
                        )
                        # Fall back to local selection
                        if self.selected_features is not None:
                            global_mask = self.selected_features
                        else:
                            # No selection available, use all features
                            global_mask = np.ones(self.X_train.shape[1], dtype=bool)

                    self.global_feature_mask = global_mask
                    X_train_selected = self.X_train[:, global_mask]
                    X_val_selected = self.X_val[:, global_mask]
                else:
                    # Empty parameters, use local selection
                    if self.selected_features is not None:
                        X_train_selected = self.X_train[:, self.selected_features]
                        X_val_selected = self.X_val[:, self.selected_features]
                    else:
                        # No selection yet, use all features
                        X_train_selected = self.X_train
                        X_val_selected = self.X_val
            else:
                # No parameters from server, use local selection
                if self.selected_features is not None:
                    X_train_selected = self.X_train[:, self.selected_features]
                    X_val_selected = self.X_val[:, self.selected_features]
                else:
                    X_train_selected = self.X_train
                    X_val_selected = self.X_val

            # Evaluate
            score = self.selector.evaluate_model(
                X_train_selected, self.y_train, X_val_selected, self.y_val
            )

            # Calculate loss (1 - score for maximization metrics)
            loss = 1.0 - score

            metrics = {
                "accuracy": float(score),
                "num_features": int(X_train_selected.shape[1]),
            }

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=float(loss),
                num_examples=len(self.X_val),
                metrics=metrics,
            )

        except Exception as e:
            logger.error(f"Client {self.client_id} evaluation failed: {e}")
            return EvaluateRes(
                status=Status(code=Code.OK, message=f"Error: {str(e)}"),
                loss=1.0,
                num_examples=len(self.X_val),
                metrics={"accuracy": 0.0, "error": str(e)},
            )


def client_fn(context: Context) -> Client:
    """
    Constructs the Feature Election Client.

    This function is called by Flower to create client instances.
    Data loading and client configuration should be done here.
    """
    # Get client configuration
    partition_id = int(context.node_config["partition-id"])

    # Get run configuration
    run_config = context.run_config
    fs_method = run_config.get("fs-method", "lasso")
    eval_metric = run_config.get("eval-metric", "f1")
    num_clients = run_config.get("num-clients", 10)

    # Load data for this client
    X_train, y_train, X_val, y_val, feature_names = load_client_data(
        client_id=partition_id,
        num_clients=num_clients,
    )

    logger.info(f"Loaded data for client {partition_id}")

    # Create and return client
    return FeatureElectionClient(
        client_id=partition_id,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        fs_method=fs_method,
        eval_metric=eval_metric,
    )


# Create the ClientApp
app = ClientApp(client_fn=client_fn)
