"""
Feature Election Client for Flower
"""

import logging
from typing import List, Optional, cast

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.record import Array
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

from .feature_election_utils import FeatureSelector
from .task import load_client_data

logger = logging.getLogger(__name__)

# Flower ClientApp
app = ClientApp()

# Storage keys for context.state
SELECTED_FEATURES_KEY = "selected_features"
FEATURE_SCORES_KEY = "feature_scores"
GLOBAL_MASK_KEY = "global_mask"


def save_selection_to_state(
    state: RecordDict,
    selected_features: np.ndarray,
    feature_scores: np.ndarray,
) -> None:
    mask_arr = ArrayRecord()
    mask_arr["mask"] = Array(selected_features.astype(np.float32))
    state[SELECTED_FEATURES_KEY] = mask_arr

    scores_arr = ArrayRecord()
    scores_arr["scores"] = Array(feature_scores.astype(np.float32))
    state[FEATURE_SCORES_KEY] = scores_arr


def save_global_mask_to_state(state: RecordDict, global_mask: np.ndarray) -> None:
    mask_arr = ArrayRecord()
    mask_arr["mask"] = Array(global_mask.astype(np.float32))
    state[GLOBAL_MASK_KEY] = mask_arr


def load_global_mask_from_state(state: RecordDict) -> Optional[np.ndarray]:
    if GLOBAL_MASK_KEY in state:
        mask_arr = state[GLOBAL_MASK_KEY]
        # Mypy requires casting before accessing specific types from a RecordDict
        if isinstance(mask_arr, dict) and "mask" in mask_arr:
            # We assume it's an ArrayRecord-like structure, but state stores values.
            # If state stores ArrayRecord directly:
            arr_val = mask_arr["mask"]
            if isinstance(arr_val, Array):
                return arr_val.numpy().astype(bool)
    return None


def get_model_weights(model: LogisticRegression) -> List[np.ndarray]:
    return [model.coef_.astype(np.float32), model.intercept_.astype(np.float32)]


def set_model_weights(model: LogisticRegression, weights: List[np.ndarray]) -> None:
    if len(weights) != 2:
        return
    model.coef_ = weights[0]
    model.intercept_ = weights[1]


@app.train()
def train(msg: Message, context: Context) -> Message:
    # Get configuration
    partition_id = int(str(context.node_config["partition-id"]))
    num_partitions = int(str(context.node_config["num-partitions"]))
    run_config = context.run_config
    fs_method = str(run_config.get("fs-method", "lasso"))
    eval_metric = str(run_config.get("eval-metric", "f1"))

    # Get message config
    config = cast(ConfigRecord, msg.content.get("config", ConfigRecord()))
    server_round = int(str(config.get("server_round", 1)))
    phase = str(config.get("phase", "feature_selection"))

    logger.info(f"Client {partition_id} | Round {server_round} | Phase: {phase}")

    if phase == "feature_selection":
        return _handle_feature_selection(
            msg, context, partition_id, num_partitions, fs_method, eval_metric
        )
    elif phase == "tuning_eval":
        return _handle_tuning_eval(msg, context, partition_id, num_partitions)
    else:
        return _handle_fl_training(msg, context, partition_id, num_partitions)


def _handle_feature_selection(
    msg: Message,
    context: Context,
    partition_id: int,
    num_partitions: int,
    fs_method: str,
    eval_metric: str,
) -> Message:
    try:
        X_train, y_train, X_val, y_val, _ = load_client_data(partition_id, num_partitions)
        selector = FeatureSelector(fs_method=fs_method, eval_metric=eval_metric)

        # Select
        selected_mask, feature_scores = selector.select_features(X_train, y_train)

        # Evaluate
        initial_score = selector.evaluate_model(X_train, y_train, X_val, y_val)
        X_train_sel = X_train[:, selected_mask]
        X_val_sel = X_val[:, selected_mask]
        fs_score = selector.evaluate_model(X_train_sel, y_train, X_val_sel, y_val)

        # Save to state
        save_selection_to_state(context.state, selected_mask, feature_scores)

        n_selected = int(np.sum(selected_mask))
        logger.info(f"Client {partition_id}: Selected {n_selected} features")

        arrays = ArrayRecord()
        arrays["feature_mask"] = Array(selected_mask.astype(np.float32))
        arrays["feature_scores"] = Array(feature_scores.astype(np.float32))

        metrics = MetricRecord(
            {
                "initial_score": float(initial_score),
                "fs_score": float(fs_score),
                "num_selected": n_selected,
                "num-examples": len(X_train),
            }
        )

        return Message(content=RecordDict({"arrays": arrays, "metrics": metrics}), reply_to=msg)
    except Exception as e:
        logger.error(f"Selection failed: {e}")
        return Message(content=RecordDict(), reply_to=msg)


def _handle_tuning_eval(
    msg: Message, context: Context, partition_id: int, num_partitions: int
) -> Message:
    """
    Quickly train a model on the provided mask and return validation accuracy.
    Used by server to auto-tune freedom_degree.
    """
    try:
        X_train, y_train, X_val, y_val, _ = load_client_data(partition_id, num_partitions)

        # Get mask from server
        arrays = msg.content.get("arrays", ArrayRecord())
        if "feature_mask" not in arrays:
            raise ValueError("No mask provided for tuning eval")

        # FIX: Explicit cast to Array to satisfy Mypy
        mask_array = cast(Array, arrays["feature_mask"])
        mask = mask_array.numpy().astype(bool)

        # Apply mask
        X_train_sel = X_train[:, mask]
        X_val_sel = X_val[:, mask]

        # Scale Data (Crucial for convergence)
        scaler = StandardScaler()
        X_train_sel = scaler.fit_transform(X_train_sel)
        X_val_sel = scaler.transform(X_val_sel)

        # Quick Train (lower iterations for speed during tuning)
        model = LogisticRegression(max_iter=200, solver="lbfgs", random_state=42)
        model.fit(X_train_sel, y_train)

        # Score
        val_score = model.score(X_val_sel, y_val)

        # Return ONLY metrics
        metrics = MetricRecord({"val_accuracy": float(val_score), "num-examples": len(X_train)})

        return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

    except Exception as e:
        logger.error(f"Tuning eval failed: {e}")
        return Message(content=RecordDict(), reply_to=msg)


def _handle_fl_training(
    msg: Message, context: Context, partition_id: int, num_partitions: int
) -> Message:
    try:
        X_train, y_train, X_val, y_val, _ = load_client_data(partition_id, num_partitions)

        # 1. Retrieve Global Mask
        content = msg.content
        arrays = content.get("arrays", ArrayRecord())

        global_mask = None
        if "feature_mask" in arrays:
            # FIX: Explicit cast
            mask_array = cast(Array, arrays["feature_mask"])
            global_mask = mask_array.numpy().astype(bool)
            save_global_mask_to_state(context.state, global_mask)
        else:
            global_mask = load_global_mask_from_state(context.state)

        # If no mask is found (e.g. skipped election), use ALL features
        if global_mask is None:
            logger.info(
                f"Client {partition_id}: No global mask found (likely skipped election). Defaulting to ALL features."
            )
            global_mask = np.ones(X_train.shape[1], dtype=bool)

        # 2. Apply Mask
        X_train_sel = X_train[:, global_mask]
        X_val_sel = X_val[:, global_mask]

        # 3. Scale Data (Important!)
        scaler = StandardScaler()
        X_train_sel = scaler.fit_transform(X_train_sel)
        X_val_sel = scaler.transform(X_val_sel)

        # 4. Initialize Model (High max_iter for FL)
        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)

        # 5. Load Global Weights if present
        if "model_weights" in arrays:
            weights_list = []
            i = 0
            while f"weight_{i}" in arrays:
                # FIX: Explicit cast
                w_arr = cast(Array, arrays[f"weight_{i}"])
                weights_list.append(w_arr.numpy())
                i += 1

            if len(weights_list) == 2:
                # We must fit with dummy data or set classes_ to initialize structure before setting coef_
                model.fit(X_train_sel[:10], y_train[:10])
                set_model_weights(model, weights_list)
                logger.info(f"Client {partition_id}: Initialized with global weights")

        # 6. Train
        if not hasattr(model, "coef_"):
            model.fit(X_train_sel, y_train)
        else:
            # If initialized, fit continues optimization
            model.fit(X_train_sel, y_train)

        # 7. Evaluate
        train_score = model.score(X_train_sel, y_train)
        val_score = model.score(X_val_sel, y_val)

        # 8. Package Response
        weights = get_model_weights(model)
        res_arrays = ArrayRecord()
        for i, w in enumerate(weights):
            res_arrays[f"weight_{i}"] = Array(w)
        res_arrays["model_weights"] = Array(np.array([1.0], dtype=np.float32))

        metrics = MetricRecord(
            {
                "train_accuracy": float(train_score),
                "val_accuracy": float(val_score),
                "train_loss": float(1.0 - train_score),
                "num-examples": len(X_train),
            }
        )

        return Message(content=RecordDict({"arrays": res_arrays, "metrics": metrics}), reply_to=msg)

    except Exception as e:
        logger.error(f"FL Training failed: {e}")
        import traceback

        traceback.print_exc()
        return Message(content=RecordDict(), reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    partition_id = int(str(context.node_config["partition-id"]))
    num_partitions = int(str(context.node_config["num-partitions"]))

    try:
        X_train, y_train, X_val, y_val, _ = load_client_data(partition_id, num_partitions)
        arrays = msg.content.get("arrays", ArrayRecord())

        # Determine mask
        global_mask = None
        if "feature_mask" in arrays:
            # FIX: Explicit cast
            mask_array = cast(Array, arrays["feature_mask"])
            global_mask = mask_array.numpy().astype(bool)

        if global_mask is None:
            global_mask = load_global_mask_from_state(context.state)

        # Fallback to all features if no mask exists
        if global_mask is None:
            global_mask = np.ones(X_train.shape[1], dtype=bool)

        X_train_sel = X_train[:, global_mask]
        X_val_sel = X_val[:, global_mask]

        # Apply Scaling
        scaler = StandardScaler()
        X_train_sel = scaler.fit_transform(X_train_sel)
        X_val_sel = scaler.transform(X_val_sel)

        if "model_weights" in arrays:
            # Full FL Evaluation
            model = LogisticRegression(max_iter=100, random_state=42)
            model.fit(X_train_sel[:10], y_train[:10])  # Init shape

            weights_list = []
            i = 0
            while f"weight_{i}" in arrays:
                # FIX: Explicit cast
                w_arr = cast(Array, arrays[f"weight_{i}"])
                weights_list.append(w_arr.numpy())
                i += 1

            if len(weights_list) == 2:
                set_model_weights(model, weights_list)

            score = model.score(X_val_sel, y_val)
            loss = 1.0 - score
        else:
            # Feature Selection Evaluation only (no global model yet)
            selector = FeatureSelector(eval_metric="f1")
            score = selector.evaluate_model(X_train_sel, y_train, X_val_sel, y_val)
            loss = 1.0 - score

        metrics = MetricRecord(
            {
                "eval_loss": float(loss),
                "eval_accuracy": float(score),
                "num-examples": len(X_val),
            }
        )
        return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return Message(content=RecordDict(), reply_to=msg)
