import xgboost as xgb
import numpy as np
import json
import os
import math
import logging
from typing import Dict, List, Literal, Optional, Tuple
import pandas as pd

from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

logger = logging.getLogger(__name__)


# ============================================================
# Similarity-based Federation: tree-level helper functions
# ============================================================

def _get_tree_list(model_json: Dict) -> List[Dict]:
    return model_json["learner"]["gradient_booster"]["model"]["trees"]

def _get_feature_names(model_json: Dict) -> List[str]:
    return model_json["learner"].get("feature_names", [])

def _get_base_score(model_json: Dict) -> float:
    bs = model_json["learner"]["learner_model_param"].get("base_score", "0.5")
    try:
        return float(bs)
    except Exception:
        return 0.5

def _safe_norm(v: np.ndarray) -> float:
    n = float(np.linalg.norm(v))
    return n if n > 1e-12 else 1e-12

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a @ b) / (_safe_norm(a) * _safe_norm(b)))

def _l1_weighted(a: np.ndarray, b: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    d = np.abs(a - b)
    if w is not None:
        d = d * w
    return float(d.sum())

def _leaf_mask(left_children: List[int], right_children: List[int]) -> np.ndarray:
    lc = np.array(left_children, dtype=int)
    rc = np.array(right_children, dtype=int)
    return (lc == -1) & (rc == -1)

def _depth_stats(left_children: List[int], right_children: List[int]) -> Tuple[float, int]:
    n = len(left_children)
    depth = np.zeros(n, dtype=int)
    stack = [0]
    visited = {0}
    while stack:
        u = stack.pop()
        for v in (left_children[u], right_children[u]):
            if v != -1 and v not in visited:
                depth[v] = depth[u] + 1
                visited.add(v)
                stack.append(v)
    return float(depth.mean()), int(depth.max())

def _feature_histogram(split_indices: List[int], n_features: int) -> np.ndarray:
    idx = np.array(split_indices, dtype=int)
    idx = idx[idx >= 0]
    hist = np.bincount(idx, minlength=n_features).astype(float)
    s = hist.sum()
    return hist / s if s > 0 else hist

def _per_feature_threshold_stats(split_indices: List[int], split_conditions: List[float], n_features: int) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.array(split_indices, dtype=int)
    cond = np.array(split_conditions, dtype=float)
    mu = np.zeros(n_features, dtype=float)
    sd = np.zeros(n_features, dtype=float)
    for f in range(n_features):
        vs = cond[idx == f]
        if len(vs) > 0:
            mu[f] = float(vs.mean())
            sd[f] = float(vs.std())
    return mu, sd

def _embed_tree(tree: Dict, n_features: int) -> Dict[str, np.ndarray]:
    left = tree["left_children"]
    right = tree["right_children"]
    split_idx = tree["split_indices"]
    split_cond = tree["split_conditions"]
    loss_changes = np.array(tree["loss_changes"], dtype=float)
    sum_hessian = np.array(tree["sum_hessian"], dtype=float)
    base_weights = np.array(tree["base_weights"], dtype=float)
    tp = tree.get("tree_param", {})
    num_nodes = int(tp.get("num_nodes", len(split_idx)))

    d_mean, d_max = _depth_stats(left, right)
    root_feat = int(split_idx[0]) if len(split_idx) > 0 else -1
    root_thr  = float(split_cond[0]) if len(split_cond) > 0 else 0.0
    hist = _feature_histogram(split_idx, n_features)
    thr_mu, thr_sd = _per_feature_threshold_stats(split_idx, split_cond, n_features)

    sig = np.array([
        float(loss_changes.mean() if len(loss_changes) else 0.0),
        float(loss_changes.std()  if len(loss_changes) else 0.0),
        float(loss_changes.max()  if len(loss_changes) else 0.0),
        float(sum_hessian.mean()  if len(sum_hessian) else 0.0),
        float(sum_hessian.std()   if len(sum_hessian) else 0.0),
        float(sum_hessian.max()   if len(sum_hessian) else 0.0),
    ], dtype=float)

    is_leaf = _leaf_mask(left, right)
    leaf_w = base_weights[is_leaf] if is_leaf.any() else np.array([], dtype=float)
    leaf_h = sum_hessian[is_leaf]  if is_leaf.any() else np.array([], dtype=float)

    return {
        "num_nodes": np.array([num_nodes], dtype=float),
        "depth_mean": np.array([d_mean], dtype=float),
        "depth_max":  np.array([d_max],  dtype=float),
        "root_feat":  np.array([root_feat], dtype=float),
        "root_thr":   np.array([root_thr],  dtype=float),
        "hist":       hist,
        "thr_mu":     thr_mu,
        "thr_sd":     thr_sd,
        "sig":        sig,
        "leaf_weights":  leaf_w,
        "leaf_hessian":  leaf_h,
    }

def _embed_model(model_json: Dict) -> List[Dict[str, np.ndarray]]:
    trees = _get_tree_list(model_json)
    n_features = int(trees[0]["tree_param"].get("num_feature", "0")) if trees else 0
    return [_embed_tree(t, n_features) for t in trees]

def _structural_similarity(e1: Dict, e2: Dict, root_bonus: bool = True,
                            root_weight: float = 0.2, root_thr_penalty: float = 0.1) -> float:
    s_hist = _cosine(e1["hist"], e2["hist"])
    pn  = -0.05 * abs(e1["num_nodes"][0]  - e2["num_nodes"][0])
    pdm = -0.05 * abs(e1["depth_mean"][0] - e2["depth_mean"][0])
    pdx = -0.05 * abs(e1["depth_max"][0]  - e2["depth_max"][0])
    bonus = 0.0
    if root_bonus and int(e1["root_feat"][0]) == int(e2["root_feat"][0]):
        bonus += root_weight
        bonus += -root_thr_penalty * abs(e1["root_thr"][0] - e2["root_thr"][0])
    return float(s_hist + pn + pdm + pdx + bonus)

def _threshold_similarity(e1: Dict, e2: Dict) -> float:
    w = (e1["hist"] + e2["hist"]) / 2.0
    d_mu = _l1_weighted(e1["thr_mu"], e2["thr_mu"], w=w)
    d_sd = _l1_weighted(e1["thr_sd"], e2["thr_sd"], w=w)
    return float(-(d_mu + d_sd))

def _signal_similarity(e1: Dict, e2: Dict) -> float:
    return _cosine(e1["sig"], e2["sig"])

def _data_proxy_similarity(e1: Dict, e2: Dict) -> float:
    s1 = 0.0
    if len(e1["leaf_weights"]) > 0 and len(e2["leaf_weights"]) > 0:
        a = np.sort(e1["leaf_weights"]); b = np.sort(e2["leaf_weights"])
        m = min(len(a), len(b))
        if m > 0:
            s1 = _cosine(a[:m], b[:m])
    s2 = 0.0
    if len(e1["leaf_hessian"]) > 0 and len(e2["leaf_hessian"]) > 0:
        a = np.sort(e1["leaf_hessian"]); b = np.sort(e2["leaf_hessian"])
        m = min(len(a), len(b))
        if m > 0:
            s2 = _cosine(a[:m], b[:m])
    return float(0.5 * (s1 + s2))

def _metadata_similarity(meta_i: Dict, meta_j: Dict) -> float:
    
    score_parts: List[Tuple[str, float, float]] = []  # (name, score, weight)

    lr_i = meta_i.get("label_ratio_count", {})
    lr_j = meta_j.get("label_ratio_count", {})
    pos_i = float(lr_i.get("1", {}).get("ratio", 0.0))
    pos_j = float(lr_j.get("1", {}).get("ratio", 0.0))
    if pos_i > 0 and pos_j > 0:
        log_i = math.log(pos_i + 1e-12)
        log_j = math.log(pos_j + 1e-12)
        label_sim = 1.0 - abs(log_i - log_j) / (abs(log_i) + abs(log_j) + 1e-12)
        label_sim = float(np.clip(label_sim, 0.0, 1.0))
    elif pos_i == 0 and pos_j == 0:
        label_sim = 1.0
    else:
        label_sim = 0.0
    score_parts.append(("label_ratio", label_sim, 0.3))

    corr_map = meta_i.get("label_correlation", {})

    num_i = meta_i.get("feature_statistics", {}).get("numerical", {})
    num_j = meta_j.get("feature_statistics", {}).get("numerical", {})
    common_num = set(num_i.keys()) & set(num_j.keys()) - {"fraud_label"}
    if common_num:
        weighted_sims, total_corr = 0.0, 0.0
        for feat in common_num:
            fi = num_i[feat]; fj = num_j[feat]
            mi, si = float(fi.get("mean", 0.0)), abs(float(fi.get("std", 0.0)))
            mj, sj = float(fj.get("mean", 0.0)), abs(float(fj.get("std", 0.0)))
            denom = (si + sj) / 2.0 + 1e-9
            d_mean = abs(mi - mj) / denom
            d_std  = abs(si - sj) / denom
            feat_sim = 1.0 / (1.0 + d_mean + d_std)
            corr_entry = corr_map.get(feat, {})
            w = abs(float(corr_entry.get("correlation", 0.0))) if isinstance(corr_entry, dict) else 0.0
            weighted_sims += feat_sim * w
            total_corr    += w
        num_sim = (weighted_sims / total_corr) if total_corr > 1e-12 else float(
            np.mean([1.0 / (1.0 + abs(float(num_i[f].get("mean", 0.0)) - float(num_j[f].get("mean", 0.0))) /
                     ((abs(float(num_i[f].get("std", 0.0))) + abs(float(num_j[f].get("std", 0.0)))) / 2.0 + 1e-9))
                     for f in common_num])
        )
        score_parts.append(("numerical", float(np.clip(num_sim, 0.0, 1.0)), 0.4))

    cat_i = meta_i.get("feature_statistics", {}).get("categorical", {})
    cat_j = meta_j.get("feature_statistics", {}).get("categorical", {})
    common_cat = set(cat_i.keys()) & set(cat_j.keys())
    if common_cat:
        weighted_sims, total_corr = 0.0, 0.0
        for feat in common_cat:
            di = cat_i[feat]; dj = cat_j[feat]
            all_cats = sorted(set(di.keys()) | set(dj.keys()))
            pi = np.array([float(di.get(c, 0.0)) for c in all_cats])
            pj = np.array([float(dj.get(c, 0.0)) for c in all_cats])
            tv = float(0.5 * np.abs(pi - pj).sum())  # Total Variation Distance
            feat_sim = 1.0 - tv
            corr_entry = corr_map.get(feat, {})
            w = abs(float(corr_entry.get("correlation", 0.0))) if isinstance(corr_entry, dict) else 0.0
            weighted_sims += feat_sim * w
            total_corr    += w
        cat_sim = (weighted_sims / total_corr) if total_corr > 1e-12 else float(
            np.mean([1.0 - 0.5 * np.abs(
                np.array([float(cat_i[f].get(c, 0.0)) for c in sorted(set(cat_i[f]) | set(cat_j[f]))]) -
                np.array([float(cat_j[f].get(c, 0.0)) for c in sorted(set(cat_i[f]) | set(cat_j[f]))])
            ).sum() for f in common_cat])
        )
        score_parts.append(("categorical", float(np.clip(cat_sim, 0.0, 1.0)), 0.3))

    if not score_parts:
        return 0.0
    total_w = sum(w for _, _, w in score_parts)
    return float(sum(s * w for _, s, w in score_parts) / (total_w + 1e-12))

def _compute_component_scores(
    target_embeds: List[Dict], foreign_embed: Dict,
    components: List[str], pool: str, pool_topk: int,
    root_bonus: bool, root_weight: float, root_thr_penalty: float
) -> Dict[str, float]:
    per_target = []
    for e in target_embeds:
        sc = {}
        if "structural"  in components: sc["structural"]  = _structural_similarity(e, foreign_embed, root_bonus, root_weight, root_thr_penalty)
        if "threshold"   in components: sc["threshold"]   = _threshold_similarity(e, foreign_embed)
        if "signal"      in components: sc["signal"]      = _signal_similarity(e, foreign_embed)
        if "data_proxy"  in components: sc["data_proxy"]  = _data_proxy_similarity(e, foreign_embed)
        per_target.append(sc)

    pooled = {}
    for c in components:
        arr = np.array([sc.get(c, 0.0) for sc in per_target], dtype=float)
        if arr.size == 0:
            pooled[c] = 0.0
        elif pool == "mean":
            pooled[c] = float(arr.mean())
        elif pool == "topk-mean":
            k = min(pool_topk, arr.size)
            pooled[c] = float(np.sort(arr)[-k:].mean()) if k > 0 else 0.0
        else:  # "max"
            pooled[c] = float(arr.max())
    return pooled

def _composite_score(scores: Dict[str, float], components: List[str], weights: List[float]) -> float:
    w = np.array(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    return float(sum(w[i] * scores.get(c, 0.0) for i, c in enumerate(components)))

def _merge_trees_into_target(target_json: Dict, chosen_trees: List[Dict], tree_info_value: int = 0) -> Dict:
    import copy
    out = json.loads(json.dumps(target_json))
    out.pop("_path", None)

    model = out["learner"]["gradient_booster"]["model"]
    trees = model["trees"]

    trees.extend(copy.deepcopy(chosen_trees))
    new_num = len(trees)

    for i, tree in enumerate(trees):
        tree["id"] = i

    orig_info = model.get("tree_info", [])
    model["tree_info"] = list(orig_info) + [tree_info_value] * len(chosen_trees)

    if "gbtree_model_param" in model:
        p = model["gbtree_model_param"]
        p["num_trees"] = str(new_num)
        p["num_roots"] = "1"
        p["size_leaf_vector"] = "1"
        nf = trees[0]["tree_param"].get("num_feature")
        if nf is not None:
            p["num_feature"] = str(nf)

    model["iteration_indptr"] = list(range(new_num + 1))
    return out

def _predict_tree(tree: Dict, X: np.ndarray) -> np.ndarray:
    left = tree["left_children"]; right = tree["right_children"]
    split_idx = tree["split_indices"]; split_cond = tree["split_conditions"]
    default_left = tree["default_left"]; base_weights = tree["base_weights"]
    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        node = 0
        while True:
            lc = left[node]; rc = right[node]
            if lc == -1 and rc == -1:
                out[i] = float(base_weights[node]); break
            feat_id = split_idx[node]; thr = split_cond[node]
            x = X[i, feat_id]
            if np.isnan(x): node = lc if bool(default_left[node]) else rc
            else:           node = lc if x < thr else rc
    return out

def _predict_model_json(model_json: Dict, X: np.ndarray) -> np.ndarray:
    trees = _get_tree_list(model_json)
    bs = _get_base_score(model_json)
    base_logit = math.log((bs + 1e-9) / (1.0 - bs + 1e-9))
    logits = np.full(X.shape[0], base_logit, dtype=float)
    for t in trees:
        logits += _predict_tree(t, X)
    return 1.0 / (1.0 + np.exp(-logits))

def _best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float, float]:
    order = np.argsort(-y_prob)
    y = y_true[order]; probs = y_prob[order]
    tp = 0; fp = 0; fn = int(y.sum())
    best = (0.0, 0.5, 0.0, 0.0)
    for i in range(len(y)):
        if y[i] == 1: tp += 1; fn -= 1
        else: fp += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
        if f1 > best[0]:
            best = (f1, float(probs[i]), prec, rec)
    return best

class FedXGBBagging:

    def __init__(
        self,
        model_paths: List[str],
        voting: Literal['soft', 'hard', 'weighted_soft'] = 'soft',
        model_weights: Optional[List[float]] = None,
        config=None,
        result_path: Optional[str] = None,
    ):
        self.model_paths = model_paths
        self.voting = voting
        self.model_weights = model_weights
        self.config = config or {}
        self.result_path = result_path
        self.history = []
        self.test_data = None

        self.models = [self._load_model(path) for path in model_paths]

        bank_name_round_number = self.config.get("bank_name_round_number", "unknown")

        logger.info(f"Initialized EachBankModel for bank_name_round_number={self.config['bank_name_round_number']}")

    def _load_model(self, path: str):
        logger.info("Loading each model")
        model = xgb.Booster()

        try:
            model.load_model(path)
        except xgb.core.XGBoostError as e:
            logging.error("XGBoostError: Failed to load the model.")
            logging.error(f"Model path: {path}")
            logging.error("Ensure the model was saved in JSON format using booster.save_model('model.json').")
            raise e
        except UnicodeDecodeError as e:
            logging.error("UnicodeDecodeError: The model file is not in valid UTF-8 encoding.")
            logging.error(f"Model path: {path}")
            logging.error("Although the file has a .json extension, it may have been saved in XGBoost's binary format.")
            logging.error("Hint: Reload the binary model and re-save it using booster.save_model('model.json').")
            raise e
        except Exception as e:
            logging.error("Unexpected error while loading the model.")
            logging.error(f"Model path: {path}")
            logging.error(f"Error message: {str(e)}")
            raise e

        return model

    def predict(
        self,
        X: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y_true : Optional[np.ndarray]
            True labels (binary). If provided and ``threshold`` is None, the method will
            compute the best F1 threshold using ``_best_f1_threshold``.
        threshold : Optional[float]
            Decision threshold for binary classification. If None, defaults to 0.5 unless
            ``y_true`` is provided (then uses the best F1 threshold on ``y_true`` / probabilities).
        """
        logger.info("Perform prediction")
        X = X.copy()
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")

        all_preds = []
        for idx, model in enumerate(self.models):
            try:
                expected_features = model.feature_names
                # ìì ë§ì¶ê³ , ëë½ì 0ì¼ë¡ ì±ì
                X_aligned = X.reindex(columns=expected_features, fill_value=0)

                dmatrix = xgb.DMatrix(X_aligned, enable_categorical=True)
                print(f"Predicting with model {idx+1}/{len(self.models)}")
                preds = model.predict(dmatrix)
                all_preds.append(preds)

            except ValueError as e:
                if "feature_names mismatch" in str(e):
                    training_features = model.feature_names or []
                    test_features = list(X.columns)
                    missing_in_test = sorted(set(training_features) - set(test_features))
                    extra_in_test = sorted(set(test_features) - set(training_features))

                    logging.error("!!!!!!!! [Feature Mismatch Detected] !!!!!!!!")
                    logging.error(f"ëª¨ë¸ {idx+1}/{len(self.models)} ìì mismatch ë°ì")
                    logging.error(f"Missing in test: {missing_in_test}")
                    logging.error(f"Extra in test: {extra_in_test}")
                    raise
                else:
                    raise

        all_preds = np.array(all_preds)

        if self.voting == 'soft':
            avg_probs = np.mean(all_preds, axis=0)
            if len(avg_probs.shape) == 1:  # binary
                if threshold is None:
                    if y_true is not None:
                        _, threshold, _, _ = _best_f1_threshold(y_true, avg_probs)
                    else:
                        threshold = 0.5
                y_pred = (avg_probs > threshold).astype(int)
            else:  # multiclass
                y_pred = np.argmax(avg_probs, axis=1)
            return y_pred, avg_probs

        elif self.voting == 'weighted_soft':
            if self.model_weights is None or len(self.model_weights) != len(self.models):
                raise ValueError("Model weights must be provided for 'weighted_soft' voting and match number of models.")
            weights = np.array(self.model_weights)
            weighted_avg = np.average(all_preds, axis=0, weights=weights)
            if len(weighted_avg.shape) == 1:
                if threshold is None:
                    if y_true is not None:
                        _, threshold, _, _ = _best_f1_threshold(y_true, weighted_avg)
                    else:
                        threshold = 0.5
                y_pred = (weighted_avg > threshold).astype(int)
            else:
                y_pred = np.argmax(weighted_avg, axis=1)
            return y_pred, weighted_avg

        elif self.voting == 'hard':
            if len(all_preds.shape) == 2:  # binary
                bin_preds = (all_preds > 0.5).astype(int)  # shape: (n_models, n_samples)
                # The output probabilities represent the percentage of models voting for class 1
                y_prob = np.mean(bin_preds, axis=0)
                if threshold is None:
                    if y_true is not None:
                        _, threshold, _, _ = _best_f1_threshold(y_true, y_prob)
                    else:
                        threshold = 0.5
                y_pred = (y_prob > threshold).astype(int)
                return y_pred, y_prob
            else:  # multiclass
                bin_preds = np.argmax(all_preds, axis=2)  # shape: (n_models, n_samples)

                # majority voting
                y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=bin_preds)

                n_classes = all_preds.shape[2]
                y_prob = np.apply_along_axis(
                    lambda x: np.bincount(x, minlength=n_classes) / len(x),
                    axis=0,
                    arr=bin_preds
                )  # shape: (n_samples, n_classes)

                return y_pred, y_prob

        else:
            raise ValueError("Voting must be 'soft', 'hard', or 'weighted_soft'.")


    def evaluate_predictions(self, y_true, y_pred, y_prob=None) -> dict:
        logger.info("Calculate metrics")
        try:
            if y_prob is not None:
                prob_for_auc = y_prob if y_prob.ndim == 1 else np.max(y_prob, axis=1)
            else:
                prob_for_auc = y_pred if y_pred.ndim == 1 else np.max(y_pred, axis=1)
            roc_auc = roc_auc_score(y_true, prob_for_auc)
            pr_auc = average_precision_score(y_true, prob_for_auc)
        except:
            roc_auc = None
            pr_auc = None

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "pr_auc": float(pr_auc) if pr_auc is not None else None
        }

    def analyze_detection_cases(self, y_true, y_pred, y_prob, output_path=None, top_k_diff_features=20):
        logger.info("Analyzing detection success and failure cases")

        test_df = self.test_data.copy()
        test_df["y_true"] = y_true
        test_df["y_pred"] = y_pred
        test_df["y_prob"] = y_prob

        def label_group(row):
            if row["y_true"] == 1 and row["y_pred"] == 1: return "TP"
            elif row["y_true"] == 1: return "FN"
            elif row["y_pred"] == 1: return "FP"
            else: return "TN"

        test_df["group"] = test_df.apply(label_group, axis=1)

        numeric_features = test_df.select_dtypes(include=["number"]).columns.difference(["y_true", "y_pred", "y_prob"])
        categorical_features = test_df.select_dtypes(include=["object", "category"]).columns

        grouped_stats = test_df.groupby("group")[numeric_features].describe()
        group_means = {
            g: test_df[test_df["group"] == g][numeric_features].mean()
            for g in ["TP", "FN", "FP", "TN"]
        }
        tp_fn_diff = (group_means["TP"] - group_means["FN"]).abs().sort_values(ascending=False).head(top_k_diff_features)
        fp_tn_diff = (group_means["FP"] - group_means["TN"]).abs().sort_values(ascending=False).head(top_k_diff_features)

        categorical_distribution = {}
        for col in categorical_features:
            col_dist = {}
            for group in ["TP", "FN", "FP", "TN"]:
                group_data = test_df[test_df["group"] == group][col].value_counts(normalize=True).to_dict()
                col_dist[group] = group_data
            categorical_distribution[col] = col_dist

        tp_fn_cat_diff, fp_tn_cat_diff = {}, {}
        for col in categorical_features:
            tp_dist = pd.Series(categorical_distribution[col].get("TP", {}))
            fn_dist = pd.Series(categorical_distribution[col].get("FN", {}))
            fp_dist = pd.Series(categorical_distribution[col].get("FP", {}))
            tn_dist = pd.Series(categorical_distribution[col].get("TN", {}))

            tp_index = set(tp_dist.index) | set(fn_dist.index)
            tp_dist = tp_dist.reindex(tp_index, fill_value=0)
            fn_dist = fn_dist.reindex(tp_index, fill_value=0)

            fp_index = set(fp_dist.index) | set(tn_dist.index)
            fp_dist = fp_dist.reindex(fp_index, fill_value=0)
            tn_dist = tn_dist.reindex(fp_index, fill_value=0)

            tp_fn_cat_diff[col] = np.abs(tp_dist - fn_dist).sum()
            fp_tn_cat_diff[col] = np.abs(fp_dist - tn_dist).sum()

        cat_diff_summary = {
            "TP_vs_FN": dict(sorted(tp_fn_cat_diff.items(), key=lambda x: x[1], reverse=True)[:top_k_diff_features]),
            "FP_vs_TN": dict(sorted(fp_tn_cat_diff.items(), key=lambda x: x[1], reverse=True)[:top_k_diff_features])
        }

        logger.info("Finished detection analysis (numeric + categorical)")
        return {
            "count_summary": dict(test_df["group"].value_counts()),
            "detection_summary": {
                "TP_vs_FN": tp_fn_diff,
                "FP_vs_TN": fp_tn_diff,
                "grouped_stats": grouped_stats,
                "categorical_distribution": categorical_distribution,
                "categorical_group_difference": cat_diff_summary
            }
        }

    def save_metrics_history(self, metrics, detection_summary, test_time=None, output_path=None):
        logger.info("Saving metrics and detection summary to history")

        def convert_to_serializable(obj):
            if isinstance(obj, (np.generic, np.int64, np.float32)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="index")
            return obj

        def flatten_grouped_stats(grouped_stats_df: pd.DataFrame) -> dict:
            result = {}
            for group in grouped_stats_df.index:
                stats_dict = {}
                for (feature, stat), value in grouped_stats_df.loc[group].items():
                    stats_dict[f"{feature}__{stat}"] = value
                result[group] = stats_dict
            return result

        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, default=convert_to_serializable)
        
        logger.info(f"Evaluation metrics: {metrics}")

        serialized_detection_summary = {
            "count_summary": detection_summary["count_summary"],
            "detection_summary": {
                "TP_vs_FN": convert_to_serializable(detection_summary["detection_summary"]["TP_vs_FN"]),
                "FP_vs_TN": convert_to_serializable(detection_summary["detection_summary"]["FP_vs_TN"]),
                "grouped_stats": convert_to_serializable(flatten_grouped_stats(detection_summary["detection_summary"]["grouped_stats"])),
                "categorical_distribution": convert_to_serializable(detection_summary["detection_summary"]["categorical_distribution"]),
                "categorical_group_difference": convert_to_serializable(detection_summary["detection_summary"]["categorical_group_difference"]),
            }
        }

        if not hasattr(self, "history"):
            self.history = []

        self.history.append({
            "configuration": convert_to_serializable(self.config),
            "metrics": metrics,
            "detection_summary": serialized_detection_summary,
            "test_time": test_time
        })

        history_path = output_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4, default=convert_to_serializable)
    

    def save_model(self, final_result_path):
        logger.info("Saving merged JSON of in-memory XGBoost models")

        #final_result_path = Path(self.config['result_path']) / f"{self.config['bank_name_round_number']}_num_clients_{self.config['n_clients']}"
        #final_result_path.mkdir(parents=True, exist_ok=True)

        model_dir = final_result_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        save_name = f"{self.config['model_name'].lower()}_model_{self.config['bank_name_round_number']}_{self.voting.lower()}.json"
        save_path = model_dir / save_name

        merged_json = {}
        for idx, booster in enumerate(self.models):
            raw_json = json.loads(booster.save_raw("json").decode("utf-8"))
            merged_json[f"model{idx+1}"] = raw_json

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(merged_json, f, indent=2)

        logger.info(f"Ensemble model (as merged JSON) saved to {save_path}")


# ============================================================
# Tree-level Similarity-based Federated Model
# ============================================================

class FedXGBSimilarity:

    VALID_COMPONENTS = {"structural", "threshold", "signal", "data_proxy", "metadata"}

    def __init__(
        self,
        target_path: str,
        other_paths: List[str],
        components: List[str] = None,
        weights: List[float] = None,
        pool: str = "max",
        pool_topk: int = 5,
        topk_per_source: int = 100,
        sim_threshold: Optional[float] = None,
        max_additional_trees: int = 100,
        root_bonus: bool = True,
        root_weight: float = 0.2,
        root_thr_penalty: float = 0.1,
        target_meta_path: Optional[str] = None,
        other_meta_paths: Optional[List[str]] = None,
        config: Optional[dict] = None,
        result_path: Optional[str] = None,
    ):
        if components is None:
            components = ["structural", "threshold", "signal", "data_proxy"]
        invalid = set(components) - self.VALID_COMPONENTS
        if invalid:
            raise ValueError(f"Unknown components: {invalid}. Choose from {self.VALID_COMPONENTS}")
        if weights is None:
            weights = [1.0 / len(components)] * len(components)
        if len(weights) != len(components):
            raise ValueError("weights length must match components length")
        if "metadata" in components:
            if target_meta_path is None or other_meta_paths is None:
                raise ValueError(
                    "'metadata' component requires target_meta_path and other_meta_paths to be provided."
                )
            if len(other_meta_paths) != len(other_paths):
                raise ValueError("other_meta_paths length must match other_paths length")

        self.target_path = target_path
        self.other_paths = other_paths
        self.components = components
        self.weights = weights
        self.pool = pool
        self.pool_topk = pool_topk
        self.topk_per_source = topk_per_source
        self.sim_threshold = sim_threshold
        self.max_additional_trees = max_additional_trees
        self.root_bonus = root_bonus
        self.root_weight = root_weight
        self.root_thr_penalty = root_thr_penalty
        self.target_meta_path = target_meta_path
        self.other_meta_paths = other_meta_paths or []
        self.config = config or {}
        self.result_path = result_path

        self.merged_json: Optional[Dict] = None   
        self._booster: Optional[xgb.Booster] = None
        self.candidate_rows: List[Dict] = []
        self.history: List[Dict] = []

        logger.info(
            f"FedXGBSimilarity initialized | target={os.path.basename(target_path)} "
            f"| sources={len(other_paths)} | components={components}"
        )

    def _load_meta(self, path: str) -> Dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _select_candidates(self, target_json: Dict, foreign_jsons: List[Dict]) -> List[Dict]:
        target_embeds = _embed_model(target_json)
        rows_all = []

        target_meta = None
        if "metadata" in self.components and self.target_meta_path:
            target_meta = self._load_meta(self.target_meta_path)
            logger.info(f"Loaded target metadata from {self.target_meta_path}")

        for src_idx, fj in enumerate(foreign_jsons):
            src_name = os.path.basename(fj.get("_path", "foreign.json"))

            meta_score: Optional[float] = None
            if "metadata" in self.components and target_meta is not None:
                src_meta_path = self.other_meta_paths[src_idx] if src_idx < len(self.other_meta_paths) else None
                if src_meta_path:
                    src_meta = self._load_meta(src_meta_path)
                    meta_score = _metadata_similarity(target_meta, src_meta)
                    logger.info(f"Metadata similarity [{src_name}]: {meta_score:.4f}")

            embeds = _embed_model(fj)
            trees = _get_tree_list(fj)

            tree_components = [c for c in self.components if c != "metadata"]
            rows = []
            for idx, (tr, emb) in enumerate(zip(trees, embeds)):
                scores = _compute_component_scores(
                    target_embeds, emb,
                    tree_components, self.pool, self.pool_topk,
                    self.root_bonus, self.root_weight, self.root_thr_penalty
                )

                if "metadata" in self.components:
                    scores["metadata"] = meta_score if meta_score is not None else 0.0
                composite = _composite_score(scores, self.components, self.weights)
                rows.append({"tree": tr, "scores": scores, "composite": composite,
                             "src": src_name, "tree_idx": idx})

            rows.sort(key=lambda r: r["composite"], reverse=True)
            if self.sim_threshold is not None:
                rows = [r for r in rows if r["composite"] >= self.sim_threshold]
            rows_all.extend(rows[:self.topk_per_source])

        rows_all.sort(key=lambda r: r["composite"], reverse=True)
        return rows_all


    def build(
        self,
        val_X: Optional[pd.DataFrame] = None,
        val_y: Optional[np.ndarray] = None,
        eta: float = 0.02,
        greedy: bool = False,
    ) -> "FedXGBSimilarity":

        with open(self.target_path, "r", encoding="utf-8") as f:
            target_json = json.load(f)
        target_json["_path"] = self.target_path

        foreign_jsons = []
        for p in self.other_paths:
            with open(p, "r", encoding="utf-8") as f:
                fj = json.load(f)
            fj["_path"] = p
            foreign_jsons.append(fj)

        logger.info("Selecting candidate trees by similarity...")
        self.candidate_rows = self._select_candidates(target_json, foreign_jsons)
        logger.info(f"Total candidates: {len(self.candidate_rows)}")

        if greedy and val_X is not None and val_y is not None:
            chosen_trees = self._greedy_forward_selection(target_json, val_X, val_y, eta)
        else:
            chosen_trees = [r["tree"] for r in self.candidate_rows[:self.max_additional_trees]]

        logger.info(f"Trees to merge: {len(chosen_trees)}")

        logger.info("Merging trees into target model JSON...")
        self.merged_json = _merge_trees_into_target(target_json, chosen_trees)
        logger.info(f"Merge done. Total trees: {len(_get_tree_list(self.merged_json))}")

        tmp_path = "_tmp_similarity_merged.json"
        try:
            logger.info("Writing merged model to temp file...")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.merged_json, f)
            logger.info("Loading merged model as XGBoost Booster...")
            booster = xgb.Booster()
            booster.load_model(tmp_path)
            self._booster = booster
            logger.info("Booster loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load merged model as Booster: {e}")
            raise
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        logger.info(
            f"Build complete | original trees={len(_get_tree_list(target_json))} "
            f"| added={len(chosen_trees)} "
            f"| total={len(_get_tree_list(self.merged_json))}"
        )
        return self

    def _greedy_forward_selection(
        self,
        target_json: Dict,
        val_X: pd.DataFrame,
        val_y: np.ndarray,
        eta: float,
    ) -> List[Dict]:
        feature_names = _get_feature_names(target_json)
        X_aligned = val_X.reindex(columns=feature_names, fill_value=0).copy()
        for col in X_aligned.select_dtypes(include=["object", "category"]).columns:
            X_aligned[col] = X_aligned[col].astype("category").cat.codes.astype(float)
        X_np = X_aligned.to_numpy(dtype=float)

        base_prob = _predict_model_json(target_json, X_np)
        best_f1, _, _, _ = _best_f1_threshold(val_y, base_prob)
        base_logits = np.log(base_prob / (1.0 - base_prob + 1e-12) + 1e-12)

        chosen = []
        cur_logits = base_logits.copy()

        for r in self.candidate_rows:
            if len(chosen) >= self.max_additional_trees:
                break
            t_pred = _predict_tree(r["tree"], X_np)
            new_logits = cur_logits + eta * t_pred
            new_prob = 1.0 / (1.0 + np.exp(-new_logits))
            new_f1, _, _, _ = _best_f1_threshold(val_y, new_prob)
            if new_f1 > best_f1:
                chosen.append(r["tree"])
                cur_logits = new_logits
                best_f1 = new_f1

        logger.info(f"Greedy selection: {len(chosen)} trees improve F1 (best_f1={best_f1:.4f})")
        return chosen

    def predict(
        self,
        X: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using the merged model.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y_true : Optional[np.ndarray]
            True labels (binary). If provided and ``threshold`` is None, the method will
            compute the best F1 threshold using ``_best_f1_threshold``.
        threshold : Optional[float]
            Decision threshold for binary classification. If None, defaults to 0.5 unless
            ``y_true`` is provided (then uses the best F1 threshold on ``y_true`` / probabilities).
        """
        if self._booster is None:
            raise RuntimeError("Model not built yet. Call build() first.")

        X = X.copy()
        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")

        feature_names = _get_feature_names(self.merged_json)
        if feature_names:
            X = X.reindex(columns=feature_names, fill_value=0)

        dmatrix = xgb.DMatrix(X, enable_categorical=True)
        y_prob = self._booster.predict(dmatrix)

        if y_prob.ndim == 1:
            if threshold is None:
                if y_true is not None:
                    _, threshold, _, _ = _best_f1_threshold(y_true, y_prob)
                else:
                    threshold = 0.5
            y_pred = (y_prob > threshold).astype(int)
        else:
            y_pred = np.argmax(y_prob, axis=1)

        return y_pred, y_prob

    def evaluate_predictions(self, y_true, y_pred, y_prob=None) -> dict:
        logger.info("Calculate metrics")
        try:
            if y_prob is not None:
                prob_for_auc = y_prob if y_prob.ndim == 1 else np.max(y_prob, axis=1)
            else:
                prob_for_auc = y_pred if y_pred.ndim == 1 else np.max(y_pred, axis=1)
            roc_auc = roc_auc_score(y_true, prob_for_auc)
            pr_auc  = average_precision_score(y_true, prob_for_auc)
        except Exception:
            roc_auc = None
            pr_auc  = None

        return {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall":    recall_score(y_true, y_pred, zero_division=0),
            "f1":        f1_score(y_true, y_pred, zero_division=0),
            "roc_auc":   float(roc_auc) if roc_auc is not None else None,
            "pr_auc":    float(pr_auc)  if pr_auc  is not None else None,
        }

    def analyze_detection_cases(self, y_true, y_pred, y_prob, output_path=None, top_k_diff_features=20):
        logger.info("Analyzing detection success and failure cases")

        test_df = self.test_data.copy()
        test_df["y_true"] = y_true
        test_df["y_pred"] = y_pred
        test_df["y_prob"] = y_prob

        def label_group(row):
            if row["y_true"] == 1 and row["y_pred"] == 1: return "TP"
            elif row["y_true"] == 1: return "FN"
            elif row["y_pred"] == 1: return "FP"
            else: return "TN"

        test_df["group"] = test_df.apply(label_group, axis=1)

        numeric_features = test_df.select_dtypes(include=["number"]).columns.difference(["y_true", "y_pred", "y_prob"])
        categorical_features = test_df.select_dtypes(include=["object", "category"]).columns

        grouped_stats = test_df.groupby("group")[numeric_features].describe()
        group_means = {
            g: test_df[test_df["group"] == g][numeric_features].mean()
            for g in ["TP", "FN", "FP", "TN"]
        }
        tp_fn_diff = (group_means["TP"] - group_means["FN"]).abs().sort_values(ascending=False).head(top_k_diff_features)
        fp_tn_diff = (group_means["FP"] - group_means["TN"]).abs().sort_values(ascending=False).head(top_k_diff_features)

        categorical_distribution = {}
        for col in categorical_features:
            col_dist = {}
            for group in ["TP", "FN", "FP", "TN"]:
                group_data = test_df[test_df["group"] == group][col].value_counts(normalize=True).to_dict()
                col_dist[group] = group_data
            categorical_distribution[col] = col_dist

        tp_fn_cat_diff, fp_tn_cat_diff = {}, {}
        for col in categorical_features:
            tp_dist = pd.Series(categorical_distribution[col].get("TP", {}))
            fn_dist = pd.Series(categorical_distribution[col].get("FN", {}))
            fp_dist = pd.Series(categorical_distribution[col].get("FP", {}))
            tn_dist = pd.Series(categorical_distribution[col].get("TN", {}))

            tp_index = set(tp_dist.index) | set(fn_dist.index)
            tp_dist = tp_dist.reindex(tp_index, fill_value=0)
            fn_dist = fn_dist.reindex(tp_index, fill_value=0)

            fp_index = set(fp_dist.index) | set(tn_dist.index)
            fp_dist = fp_dist.reindex(fp_index, fill_value=0)
            tn_dist = tn_dist.reindex(fp_index, fill_value=0)

            tp_fn_cat_diff[col] = np.abs(tp_dist - fn_dist).sum()
            fp_tn_cat_diff[col] = np.abs(fp_dist - tn_dist).sum()

        cat_diff_summary = {
            "TP_vs_FN": dict(sorted(tp_fn_cat_diff.items(), key=lambda x: x[1], reverse=True)[:top_k_diff_features]),
            "FP_vs_TN": dict(sorted(fp_tn_cat_diff.items(), key=lambda x: x[1], reverse=True)[:top_k_diff_features])
        }

        logger.info("Finished detection analysis (numeric + categorical)")
        return {
            "count_summary": dict(test_df["group"].value_counts()),
            "detection_summary": {
                "TP_vs_FN": tp_fn_diff,
                "FP_vs_TN": fp_tn_diff,
                "grouped_stats": grouped_stats,
                "categorical_distribution": categorical_distribution,
                "categorical_group_difference": cat_diff_summary
            }
        }

    def save_metrics_history(self, metrics, detection_summary, test_time=None, output_path=None):
        logger.info("Saving metrics and detection summary to history")

        def convert_to_serializable(obj):
            if isinstance(obj, (np.generic, np.int64, np.float32)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient="index")
            return obj

        def flatten_grouped_stats(grouped_stats_df: pd.DataFrame) -> dict:
            result = {}
            for group in grouped_stats_df.index:
                stats_dict = {}
                for (feature, stat), value in grouped_stats_df.loc[group].items():
                    stats_dict[f"{feature}__{stat}"] = value
                result[group] = stats_dict
            return result

        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4, default=convert_to_serializable)
        
        logger.info(f"Evaluation metrics: {metrics}")

        serialized_detection_summary = {
            "count_summary": detection_summary["count_summary"],
            "detection_summary": {
                "TP_vs_FN": convert_to_serializable(detection_summary["detection_summary"]["TP_vs_FN"]),
                "FP_vs_TN": convert_to_serializable(detection_summary["detection_summary"]["FP_vs_TN"]),
                "grouped_stats": convert_to_serializable(flatten_grouped_stats(detection_summary["detection_summary"]["grouped_stats"])),
                "categorical_distribution": convert_to_serializable(detection_summary["detection_summary"]["categorical_distribution"]),
                "categorical_group_difference": convert_to_serializable(detection_summary["detection_summary"]["categorical_group_difference"]),
            }
        }

        if not hasattr(self, "history"):
            self.history = []

        self.history.append({
            "configuration": convert_to_serializable(self.config),
            "metrics": metrics,
            "detection_summary": serialized_detection_summary,
            "test_time": test_time
        })

        history_path = output_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=4, default=convert_to_serializable)

    def save_model(self, output_path: Path) -> None:
        if self.merged_json is None:
            raise RuntimeError("No merged model to save. Call build() first.")

        model_dir = output_path / "models"
        model_dir.mkdir(parents=True, exist_ok=True)

        bank_id   = self.config.get("bank_name_round_number", "unknown")
        model_name = self.config.get("model_name", "fedxgbsimilarity").lower()
        save_path = model_dir / f"{model_name}_model_{bank_id}_similarity.json"

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self.merged_json, f, indent=2, ensure_ascii=False)

        logger.info(f"Similarity-merged model saved to {save_path}")

    def save_candidates_csv(self, output_path: Path) -> None:
        import csv
        if not self.candidate_rows:
            return
        score_cols = list(self.candidate_rows[0]["scores"].keys())
        cols = ["src", "tree_idx", "composite"] + score_cols
        csv_path = output_path / "similarity_candidates.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.candidate_rows:
                w.writerow(
                    [r["src"], r["tree_idx"], r.get("composite", 0.0)]
                    + [r["scores"].get(c, 0.0) for c in score_cols]
                )
        logger.info(f"Candidate scores saved to {csv_path}")


# ============================================================
# Ensemble-Level Similarity: Xgboost model (Ensemble-level) helper function
# ============================================================

def _compute_model_level_similarity(
    target_jsons: List[Dict],
    foreign_json: Dict,
    components: List[str],
    weights: List[float],
    pool: str,
    pool_topk: int,
    root_bonus: bool,
    root_weight: float,
    root_thr_penalty: float,
    max_trees_per_model: int = 100,
) -> Dict:

    def _get_sampled_embeds(model_json: Dict, max_trees: int) -> List[Dict]:
        trees = _get_tree_list(model_json)
        if not trees:
            return []
        n_features = int(trees[0]["tree_param"].get("num_feature", "0"))
        if len(trees) > max_trees:
            indices = np.linspace(0, len(trees) - 1, max_trees, dtype=int)
            trees = [trees[i] for i in indices]
        return [_embed_tree(t, n_features) for t in trees]

    foreign_embeds = _get_sampled_embeds(foreign_json, max_trees_per_model)
    if not foreign_embeds:
        return {"composite": 0.0, "scores": {c: 0.0 for c in components}}

    all_composites: List[float] = []
    all_component_scores: Dict[str, List[float]] = {c: [] for c in components}

    for target_json in target_jsons:
        target_embeds = _get_sampled_embeds(target_json, max_trees_per_model)
        if not target_embeds:
            continue

        tree_composites: List[float] = []
        tree_scores: Dict[str, List[float]] = {c: [] for c in components}

        for fe in foreign_embeds:
            scores = _compute_component_scores(
                target_embeds, fe,
                components, pool, pool_topk,
                root_bonus, root_weight, root_thr_penalty,
            )
            composite = _composite_score(scores, components, weights)
            tree_composites.append(composite)
            for c in components:
                tree_scores[c].append(scores.get(c, 0.0))

        if tree_composites:
            all_composites.append(float(np.mean(tree_composites)))
            for c in components:
                all_component_scores[c].append(float(np.mean(tree_scores[c])))

    if not all_composites:
        return {"composite": 0.0, "scores": {c: 0.0 for c in components}}

    return {
        "composite": float(np.mean(all_composites)),
        "scores": {c: float(np.mean(all_component_scores[c])) for c in components},
    }


# ============================================================
# Ensemble-Level Similarity Federated XGBBagging
# ============================================================

class FedEnsembleLevelSimXGBBagging:

    VALID_COMPONENTS = {"structural", "threshold", "signal", "data_proxy"}

    def __init__(
        self,
        target_paths: List[str],
        other_paths: List[str],
        components: List[str] = None,
        weights: List[float] = None,
        pool: str = "mean",
        pool_topk: int = 5,
        topk_models: int = 3,
        sim_threshold: Optional[float] = None,
        max_trees_per_model: int = 100,
        voting: str = "soft",
        model_weights: Optional[List[float]] = None,
        root_bonus: bool = True,
        root_weight: float = 0.2,
        root_thr_penalty: float = 0.1,
        config: Optional[dict] = None,
        result_path: Optional[str] = None,
    ):
        if components is None:
            components = ["structural", "threshold", "signal", "data_proxy"]
        invalid = set(components) - self.VALID_COMPONENTS
        if invalid:
            raise ValueError(f"Unknown components: {invalid}. Choose from {self.VALID_COMPONENTS}")
        if weights is None:
            weights = [1.0 / len(components)] * len(components)
        if len(weights) != len(components):
            raise ValueError("weights length must match components length")

        self.target_paths = target_paths
        self.other_paths = other_paths
        self.components = components
        self.weights = weights
        self.pool = pool
        self.pool_topk = pool_topk
        self.topk_models = topk_models
        self.sim_threshold = sim_threshold
        self.max_trees_per_model = max_trees_per_model
        self.voting = voting
        self.model_weights = model_weights
        self.root_bonus = root_bonus
        self.root_weight = root_weight
        self.root_thr_penalty = root_thr_penalty
        self.config = config or {}
        self.result_path = result_path

        self.model_similarity_rows: List[Dict] = []
        self.selected_paths: List[str] = []
        self._ensemble: Optional[FedXGBBagging] = None
        self.history: List[Dict] = []
        self.test_data = None

        logger.info(
            f"FedEnsembleLevelSimXGBBagging initialized | "
            f"targets={len(target_paths)} | candidates={len(other_paths)} | "
            f"topk_models={topk_models} | voting={voting} | components={components}"
        )

    def build(self) -> "FedEnsembleLevelSimXGBBagging":
        """ì ì¬ë ê¸°ë°ì¼ë¡ ì¸ë¶ ëª¨ë¸ì ì ííê³  ìµì¢ ììë¸ì êµ¬ì±í©ëë¤."""
        logger.info("Loading target model JSONs for similarity computation...")
        target_jsons: List[Dict] = []
        for p in self.target_paths:
            with open(p, "r", encoding="utf-8") as f:
                tj = json.load(f)
            tj["_path"] = p
            target_jsons.append(tj)
        logger.info(f"Loaded {len(target_jsons)} target model(s).")

        logger.info(f"Computing model-level similarity for {len(self.other_paths)} candidate model(s)...")
        rows: List[Dict] = []
        for p in self.other_paths:
            with open(p, "r", encoding="utf-8") as f:
                fj = json.load(f)
            src_name = os.path.basename(p)

            sim_result = _compute_model_level_similarity(
                target_jsons, fj,
                self.components, self.weights,
                self.pool, self.pool_topk,
                self.root_bonus, self.root_weight, self.root_thr_penalty,
                self.max_trees_per_model,
            )
            rows.append({
                "path": p,
                "src": src_name,
                "composite": sim_result["composite"],
                "scores": sim_result["scores"],
            })
            score_detail = " | ".join(
                f"{c}={sim_result['scores'].get(c, 0.0):.4f}" for c in self.components
            )
            logger.info(f"  [{src_name}] composite={sim_result['composite']:.4f} | {score_detail}")

        rows.sort(key=lambda r: r["composite"], reverse=True)
        self.model_similarity_rows = rows

        if self.sim_threshold is not None:
            filtered = [r for r in rows if r["composite"] >= self.sim_threshold]
            logger.info(
                f"After sim_threshold={self.sim_threshold}: "
                f"{len(filtered)}/{len(rows)} candidate(s) remain"
            )
        else:
            filtered = rows

        selected_rows = filtered[: self.topk_models]
        self.selected_paths = [r["path"] for r in selected_rows]

        logger.info(
            f"Selected {len(self.selected_paths)} foreign model(s) "
            f"from {len(self.other_paths)} candidate(s):"
        )
        for r in selected_rows:
            logger.info(f"  [SELECTED] {r['src']} (composite={r['composite']:.4f})")
        for r in rows[len(selected_rows):]:
            logger.info(f"  [EXCLUDED] {r['src']} (composite={r['composite']:.4f})")

        all_paths = list(self.target_paths) + self.selected_paths
        logger.info(
            f"Final ensemble: {len(all_paths)} model(s) "
            f"({len(self.target_paths)} target + {len(self.selected_paths)} selected foreign)"
        )

        self._ensemble = FedXGBBagging(
            model_paths=all_paths,
            voting=self.voting,
            model_weights=self.model_weights,
            config=self.config,
            result_path=self.result_path,
        )
        return self


    def predict(
        self,
        X: pd.DataFrame,
        y_true: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._ensemble is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        return self._ensemble.predict(X, y_true=y_true, threshold=threshold)

    def evaluate_predictions(self, y_true, y_pred, y_prob=None) -> dict:
        if self._ensemble is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        return self._ensemble.evaluate_predictions(y_true, y_pred, y_prob)

    def analyze_detection_cases(self, y_true, y_pred, y_prob, output_path=None, top_k_diff_features=20):
        if self._ensemble is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        self._ensemble.test_data = self.test_data
        return self._ensemble.analyze_detection_cases(
            y_true, y_pred, y_prob, output_path, top_k_diff_features
        )

    def save_metrics_history(self, metrics, detection_summary, test_time=None, output_path=None):
        if self._ensemble is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        self._ensemble.save_metrics_history(metrics, detection_summary, test_time, output_path)


    def save_model(self, final_result_path: Path) -> None:
        """ììë¸ ëª¨ë¸ì ì ì¥í©ëë¤."""
        if self._ensemble is None:
            raise RuntimeError("Model not built yet. Call build() first.")
        self._ensemble.save_model(final_result_path)

    def save_similarity_csv(self, output_path: Path) -> None:
        """ëª¨ë¸ ìì¤ ì ì¬ë ì ìë¥¼ CSVë¡ ì ì¥í©ëë¤."""
        import csv
        if not self.model_similarity_rows:
            return
        score_cols = list(self.model_similarity_rows[0]["scores"].keys())
        cols = ["src", "path", "composite", "selected"] + score_cols
        csv_path = output_path / "model_similarity_scores.csv"
        selected_set = set(self.selected_paths)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in self.model_similarity_rows:
                w.writerow(
                    [r["src"], r["path"], r["composite"], r["path"] in selected_set]
                    + [r["scores"].get(c, 0.0) for c in score_cols]
                )
        logger.info(f"Model similarity scores saved to {csv_path}")



