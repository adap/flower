"""frauddetection: XGBoost model training and data utilities."""

import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────
# Dataset constants
# ──────────────────────────────────────────────

LABEL_COL = "Fraud_Label"
CAT_COLS = [" ERC20 most sent token type", " ERC20_most_rec_token_type"]

# XGBoost params tuned for imbalanced binary fraud detection
DEFAULT_XGB_PARAMS: dict = {
    "objective": "binary:logistic",
    "eval_metric": ["auc", "logloss"],
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "scale_pos_weight": 10,  # compensate for fraud-minority class
    "tree_method": "hist",
    "seed": 42,
}


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def _cast_categoricals(X: pd.DataFrame) -> pd.DataFrame:
    """Label-encode string/categorical columns as integer codes.

    Using XGBoost's native ``category`` dtype requires the same set of
    categories in train and test, which is fragile with small partitions.
    Integer label-encoding is robust and works with standard DMatrix.
    """
    X = X.copy()
    for col in CAT_COLS:
        if col in X.columns:
            X[col] = X[col].astype(object).fillna("__missing__").astype("category").cat.codes
    return X


def preprocess_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return (X, y) from the Ethereum transaction DataFrame.

    Rows where the label is NaN are dropped (they appear in the raw CSV).
    Categorical columns are integer label-encoded.
    """
    df = df.dropna(subset=[LABEL_COL]).copy()
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].astype(int).values
    X = _cast_categoricals(X)
    return X, y


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def _split(X: pd.DataFrame, y: np.ndarray):
    """80/20 train/test split; stratified when both classes are present."""
    stratify = y if len(np.unique(y)) > 1 else None
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)


def load_sim_data(
    partition_id: int,
    num_partitions: int,
    data_csv: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Partition the CSV for simulation engine (IID split by row index)."""
    df = pd.read_csv(data_csv)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    n = len(df)
    size = n // num_partitions
    start = partition_id * size
    end = start + size if partition_id < num_partitions - 1 else n
    partition = df.iloc[start:end].reset_index(drop=True)

    X, y = preprocess_df(partition)
    X_train, X_test, y_train, y_test = _split(X, y)
    return X_train, X_test, y_train, y_test


def load_local_data(
    data_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load a local CSV file for deployment engine."""
    df = pd.read_csv(data_path)
    X, y = preprocess_df(df)
    X_train, X_test, y_train, y_test = _split(X, y)
    return X_train, X_test, y_train, y_test


# ──────────────────────────────────────────────
# Training & evaluation
# ──────────────────────────────────────────────

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    local_epochs: int = 50,
    params: dict | None = None,
) -> xgb.Booster:
    """Train an XGBoost model on local data."""
    if params is None:
        params = DEFAULT_XGB_PARAMS.copy()

    dtrain = xgb.DMatrix(_cast_categoricals(X_train), label=y_train)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=local_epochs,
        verbose_eval=False,
    )
    return booster


def evaluate_xgboost(
    booster: xgb.Booster,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
) -> Tuple[float, float]:
    """Evaluate an XGBoost model. Returns (accuracy, roc_auc)."""
    dtest = xgb.DMatrix(_cast_categoricals(X_test))
    y_prob = booster.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except ValueError:
        auc = 0.0
    return acc, auc


# ──────────────────────────────────────────────
# Model serialization (bytes ↔ XGBoost Booster)
# ──────────────────────────────────────────────

def serialize_model(booster: xgb.Booster) -> bytes:
    """Save Booster to a temp JSON file and return its raw bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    try:
        booster.save_model(tmp.name)
        with open(tmp.name, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp.name)


def deserialize_model(model_bytes: bytes) -> xgb.Booster:
    """Load an XGBoost Booster from raw JSON bytes."""
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    try:
        tmp.write(model_bytes)
        tmp.close()
        booster = xgb.Booster()
        booster.load_model(tmp.name)
    finally:
        os.unlink(tmp.name)
    return booster


def model_bytes_to_numpy(model_bytes: bytes) -> np.ndarray:
    """Encode model bytes as a uint8 numpy array for ArrayRecord storage."""
    return np.frombuffer(model_bytes, dtype=np.uint8).copy()


def numpy_to_model_bytes(arr) -> bytes:
    """Decode a uint8 numpy array (or Flower Array) back to raw bytes.

    Accepts both plain ``np.ndarray`` and Flower's ``Array``/``NDArray``
    wrappers that expose a ``.numpy()`` method.
    """
    if hasattr(arr, "numpy"):
        arr = arr.numpy()
    return np.array(arr, dtype=np.uint8).tobytes()
