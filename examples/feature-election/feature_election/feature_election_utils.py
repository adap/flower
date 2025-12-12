"""Feature Selection Utilities for Feature Election.

Provides various feature selection methods and evaluation utilities.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Type ignores added to bypass missing stub errors
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_selection import (  # type: ignore
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression  # type: ignore
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore

# Try to import PyImpetus
try:
    from PyImpetus import PPIMBC  # type: ignore

    PYIMPETUS_AVAILABLE = True
except ImportError:
    PYIMPETUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureSelector:
    """Feature selector supporting multiple methods.

    Supported methods:
    - lasso: L1-regularized linear regression
    - elastic_net: Elastic Net regularization
    - random_forest: Random Forest feature importance
    - mutual_info: Mutual information
    - chi2: Chi-squared test
    - f_classif: F-statistic
    - rfe: Recursive Feature Elimination
    - pyimpetus: PyImpetus methods (requires PyImpetus package)
    """

    def __init__(
        self,
        fs_method: str = "lasso",
        fs_params: Optional[Dict] = None,
        eval_metric: str = "f1",
        quick_eval: bool = True,
    ):
        """Initialize Feature Selector.

        Args:
            fs_method: Feature selection method
            fs_params: Parameters for the method
            eval_metric: Evaluation metric ('f1', 'accuracy', 'auc')
            quick_eval: Whether to use quick evaluation (fewer iterations)
        """
        self.fs_method = fs_method.lower()
        self.fs_params = fs_params or {}
        self.eval_metric = eval_metric
        self.quick_eval = quick_eval

        # Set default parameters
        self._set_default_params()

    def _set_default_params(self) -> None:
        """Set default parameters for each method."""
        defaults: Dict[str, Dict[str, Any]] = {
            "lasso": {"alpha": 0.01, "max_iter": 1000},
            "elastic_net": {"alpha": 0.01, "l1_ratio": 0.5, "max_iter": 1000},
            "mutual_info": {"n_neighbors": 3, "random_state": 42},
            "chi2": {"k": 10},
            "f_classif": {"k": 10},
            "rfe": {"n_features_to_select": 10, "step": 1},
            "random_forest": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "selectkbest": {"k": 10, "score_func": "f_classif"},
            "pyimpetus": {
                "model": "random_forest",
                "p_val_thresh": 0.05,
                # "num_sim": 50,
                "random_state": 42,
                "verbose": 0,
            },
        }

        if self.fs_method in defaults:
            self.fs_params = {**defaults[self.fs_method], **self.fs_params}

    def select_features(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform feature selection.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (selected_mask, feature_scores)
        """
        n_features = X.shape[1]

        # Handle PyImpetus
        if self.fs_method == "pyimpetus":
            return self._select_pyimpetus(X, y)

        # Scale data for methods that need it
        if self.fs_method in ["lasso", "elastic_net"]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X

        if self.fs_method == "lasso":
            selected_mask, feature_scores = self._select_lasso(X_scaled, y)

        elif self.fs_method == "elastic_net":
            selected_mask, feature_scores = self._select_elastic_net(X_scaled, y)

        elif self.fs_method == "mutual_info":
            selected_mask, feature_scores = self._select_mutual_info(X_scaled, y)

        elif self.fs_method == "chi2":
            selected_mask, feature_scores = self._select_chi2(X_scaled, y)

        elif self.fs_method == "f_classif":
            selected_mask, feature_scores = self._select_f_classif(X_scaled, y)

        elif self.fs_method == "rfe":
            selected_mask, feature_scores = self._select_rfe(X_scaled, y)

        elif self.fs_method == "random_forest":
            selected_mask, feature_scores = self._select_random_forest(X_scaled, y)

        elif self.fs_method == "selectkbest":
            selected_mask, feature_scores = self._select_selectkbest(X_scaled, y)

        else:
            # Default: select all features
            logger.warning(f"Unknown method {self.fs_method}, selecting all features")
            selected_mask = np.ones(n_features, dtype=bool)
            feature_scores = np.ones(n_features)

        # Ensure at least one feature is selected
        if np.sum(selected_mask) == 0:
            logger.warning("No features selected, selecting top feature")
            if len(feature_scores) > 0:
                top_feature = np.argmax(feature_scores)
                selected_mask = np.zeros(n_features, dtype=bool)
                selected_mask[top_feature] = True

        # Normalize scores to [0, 1]
        if np.max(feature_scores) > np.min(feature_scores):
            feature_scores = (feature_scores - np.min(feature_scores)) / (
                np.max(feature_scores) - np.min(feature_scores)
            )
        else:
            feature_scores = selected_mask.astype(float)

        return selected_mask, feature_scores

    def _select_lasso(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Lasso feature selection."""
        selector = Lasso(**self.fs_params)
        selector.fit(X, y)
        feature_scores = np.abs(selector.coef_)
        selected_mask = feature_scores > 1e-6
        return selected_mask, feature_scores

    def _select_elastic_net(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Elastic Net feature selection."""
        selector = ElasticNet(**self.fs_params)
        selector.fit(X, y)
        feature_scores = np.abs(selector.coef_)
        selected_mask = feature_scores > 1e-6
        return selected_mask, feature_scores

    def _select_mutual_info(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mutual information feature selection."""
        n_features = X.shape[1]
        feature_scores = mutual_info_classif(
            X,
            y,
            n_neighbors=self.fs_params.get("n_neighbors", 3),
            random_state=self.fs_params.get("random_state", 42),
        )
        k = min(self.fs_params.get("k", 10), n_features)
        selected_indices = np.argsort(feature_scores)[-k:]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[selected_indices] = True
        return selected_mask, feature_scores

    def _select_chi2(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Chi-squared feature selection."""
        n_features = X.shape[1]
        # Chi2 requires non-negative features
        X_positive = X - np.min(X, axis=0)
        feature_scores, _ = chi2(X_positive, y)
        k = min(self.fs_params.get("k", 10), n_features)
        selected_indices = np.argsort(feature_scores)[-k:]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[selected_indices] = True
        return selected_mask, feature_scores

    def _select_f_classif(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """F-statistic feature selection."""
        n_features = X.shape[1]
        feature_scores, _ = f_classif(X, y)
        k = min(self.fs_params.get("k", 10), n_features)
        selected_indices = np.argsort(feature_scores)[-k:]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[selected_indices] = True
        return selected_mask, feature_scores

    def _select_rfe(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recursive Feature Elimination."""
        n_features = X.shape[1]
        estimator = LogisticRegression(max_iter=1000, random_state=42)
        selector = RFE(
            estimator,
            n_features_to_select=min(
                self.fs_params.get("n_features_to_select", 10), n_features
            ),
            step=self.fs_params.get("step", 1),
        )
        selector.fit(X, y)
        selected_mask = selector.support_
        feature_scores = selector.ranking_.astype(float)
        # Convert ranking to scores (lower ranking = better)
        feature_scores = 1.0 / feature_scores
        return selected_mask, feature_scores

    def _select_random_forest(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Random Forest feature importance."""
        n_features = X.shape[1]
        rf = RandomForestClassifier(**self.fs_params)
        rf.fit(X, y)
        feature_scores = rf.feature_importances_
        k = min(self.fs_params.get("k", 10), n_features)
        selected_indices = np.argsort(feature_scores)[-k:]
        selected_mask = np.zeros(n_features, dtype=bool)
        selected_mask[selected_indices] = True
        return selected_mask, feature_scores

    def _select_selectkbest(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SelectKBest feature selection."""
        n_features = X.shape[1]
        score_func_name = self.fs_params.get("score_func", "f_classif")

        if score_func_name == "chi2":
            X_positive = X - np.min(X, axis=0)
            score_func = chi2
            X_to_use = X_positive
        elif score_func_name == "mutual_info":
            score_func = mutual_info_classif
            X_to_use = X
        else:
            score_func = f_classif
            X_to_use = X

        selector = SelectKBest(
            score_func=score_func, k=min(self.fs_params.get("k", 10), n_features)
        )
        selector.fit(X_to_use, y)
        selected_mask = selector.get_support()
        feature_scores = selector.scores_
        return selected_mask, feature_scores

    def _select_pyimpetus(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PyImpetus feature selection."""
        if not PYIMPETUS_AVAILABLE:
            logger.error("PyImpetus not available. Install with: pip install PyImpetus")
            # Fallback to mutual info
            return self._select_mutual_info(X, y)

        try:
            model_type = self.fs_params.get("model", "random_forest")
            p_val_thresh = self.fs_params.get("p_val_thresh", 0.05)
            # num_sim = self.fs_params.get("num_sim", 50)
            random_state = self.fs_params.get("random_state", 42)
            verbose = self.fs_params.get("verbose", 0)

            n_features = X.shape[1]

            # Initialize base model
            if model_type == "random_forest":
                base_model = RandomForestClassifier(
                    n_estimators=100, random_state=random_state, max_depth=None
                )
            elif model_type == "logistic":
                base_model = LogisticRegression(
                    max_iter=1000, random_state=random_state, solver="liblinear"
                )
            else:
                base_model = RandomForestClassifier(
                    n_estimators=100, random_state=random_state
                )

            # Use PPIMBC for feature selection
            selector = PPIMBC(
                base_model,
                p_val_thresh=p_val_thresh,
                # num_sim=num_sim,
                random_state=random_state,
                verbose=verbose,
            )
            selector.fit(X, y)

            # Get selected features from MB (Markov Blanket)
            # MB contains feature names like 'Column0', 'Column1', etc.
            selected_feature_names = selector.MB

            # Convert feature names to indices
            selected_indices = []
            for name in selected_feature_names:
                if isinstance(name, str) and name.startswith("Column"):
                    try:
                        idx = int(name.replace("Column", ""))
                        if 0 <= idx < n_features:
                            selected_indices.append(idx)
                    except ValueError:
                        continue
                elif isinstance(name, int):
                    if 0 <= name < n_features:
                        selected_indices.append(name)

            # Create binary mask
            selected_mask = np.zeros(n_features, dtype=bool)
            if len(selected_indices) > 0:
                selected_mask[selected_indices] = True
            else:
                # Fallback if no features selected
                logger.warning(
                    "PyImpetus selected no features, falling back to mutual_info"
                )
                return self._select_mutual_info(X, y)

            # Create feature scores
            # PyImpetus returns feat_imp_scores for selected features only
            feature_scores = np.zeros(n_features)
            if (
                hasattr(selector, "feat_imp_scores")
                and len(selector.feat_imp_scores) > 0
            ):
                # Assign importance scores to selected features
                for idx, score in zip(selected_indices, selector.feat_imp_scores):
                    if idx < n_features:
                        feature_scores[idx] = float(score)
            else:
                # Default: equal importance for all selected features
                feature_scores[selected_indices] = 1.0

            return selected_mask, feature_scores

        except Exception as e:
            logger.error(f"PyImpetus feature selection failed: {e}")
            return self._select_mutual_info(X, y)

    def evaluate_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Quick evaluation of model performance.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Performance score
        """
        # Skip if validation set is too small
        if len(y_val) < 5:
            return 0.5

        # Train simple model with increased max_iter to avoid convergence warnings
        max_iter = 500 if self.quick_eval else 2000
        model = LogisticRegression(max_iter=max_iter, random_state=42, solver="lbfgs")

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if self.eval_metric == "f1":
                score = f1_score(y_val, y_pred, average="weighted")
            elif self.eval_metric == "accuracy":
                score = accuracy_score(y_val, y_pred)
            elif self.eval_metric == "auc":
                if len(np.unique(y_val)) == 2:
                    y_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_proba)
                else:
                    score = f1_score(y_val, y_pred, average="weighted")
            else:
                score = f1_score(y_val, y_pred, average="weighted")

            return float(max(score, 0.0))

        except Exception as e:
            logger.warning(f"Model evaluation failed: {e}, returning default score")
            return 0.5
