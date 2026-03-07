"""xgboost_comprehensive: A Flower / XGBoost app."""

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from federated_pft_classification.data_processing.gli22_calc import GLIReferenceValueCalculator
from federated_pft_classification.data_processing.ers21_lung_volumes_calc import ERSLungVolumeCalculator
from federated_pft_classification.data_processing.fef75_calc import FEF75ValueCalculator
from federated_pft_classification.data_processing.decision_tree import interpret_pft

# Maps decision-tree output strings to integer class labels (7 classes)
LABEL_MAP = {
    "N":     0,  # normal
    "AO":    1,  # obstruction
    "R":     2,  # restriction
    "R+AO":  3,  # mixed restriction + obstruction
    "GT":    4,  # gas trapping
    "SAO":   5,  # small airway obstruction
    "R+SAO": 6,  # restriction + small airway obstruction
}

# Cached calculator instances (initialised once per process)
_ers_calc = None
_gli22_calc = None
_fef75_calc = None


def _get_calculators():
    global _ers_calc, _gli22_calc, _fef75_calc
    if _ers_calc is None:
        _ers_calc = ERSLungVolumeCalculator()
        _gli22_calc = GLIReferenceValueCalculator()
        _fef75_calc = FEF75ValueCalculator()
    return _ers_calc, _gli22_calc, _fef75_calc


def train_test_split(df, test_fraction, seed):
    """Split a DataFrame into train and validation sets."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    return train, test, len(train), len(test)


def transform_dataset_to_dmatrix(df):
    """Convert a preprocessed DataFrame to XGBoost DMatrix.

    The DataFrame must have columns:
        age, height, sex_enc, fev1, fvc, fev1_fvc   (features)
        label                                         (integer target 0-6)
    """
    feature_cols = ["age", "height", "sex_enc", "fev1", "fvc", "fev1_fvc"]
    x = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.float32)
    return xgb.DMatrix(x, label=y)


def preprocess_pft_data(
    df,
    col_age, col_height, col_sex,
    col_fev1, col_fvc, col_fev1_fvc,
    col_fef75, col_tlc, col_rv, col_rv_tlc,
):
    """Run LLN/ULN calculation and decision-tree labelling on a raw patient DataFrame.

    Returns a new DataFrame with columns:
        age, height, sex_enc, fev1, fvc, fev1_fvc, label
    Rows where the decision tree returns "Non-specific" or an error are dropped.
    """
    ers_calc, gli22_calc, fef75_calc = _get_calculators()

    records = []
    for _, row in df.iterrows():
        try:
            age          = float(row[col_age])
            height       = float(row[col_height])
            sex          = str(row[col_sex]).strip().lower()
            fev1         = float(row[col_fev1])
            fvc          = float(row[col_fvc])
            fev1_fvc_val = float(row[col_fev1_fvc])
            fef75        = float(row[col_fef75])
            tlc          = float(row[col_tlc])
            rv           = float(row[col_rv])
            rv_tlc       = float(row[col_rv_tlc])
        except (ValueError, TypeError, KeyError):
            continue  # skip rows with missing/invalid data

        if pd.isna(age) or pd.isna(height) or pd.isna(fev1) or pd.isna(fvc):
            continue
        if age <= 0 or height <= 0:
            continue

        try:
            # --- Reference value calculations ---
            tlc_res     = ers_calc.calculate_tlc(sex, height, age)
            rv_res      = ers_calc.calculate_rv(sex, height, age)
            rvtlc_res   = ers_calc.calculate_rvtlc(sex, height, age)
            fev1_res    = gli22_calc.calculate_fev1(sex, height, age)
            fvc_res     = gli22_calc.calculate_fvc(sex, height, age)
            fev1fvc_res = gli22_calc.calculate_fev1fvc(sex, height, age)
            fef75_res   = fef75_calc.calculate_fef75(sex, height, age)

            # --- Derived percent-predicted values (decision tree only) ---
            percent_pred_fvc = fvc / fvc_res["FVC Predicted"] * 100
            percent_pred_rv  = rv  / rv_res["rv Predicted"]   * 100

            # --- Run decision tree ---
            diagnosis = interpret_pft(
                fev1_fvc=fev1_fvc_val,
                tlc=tlc,
                fev1=fev1,
                fvc=fvc,
                fef75=fef75,
                percent_pred_fvc=percent_pred_fvc,
                percent_pred_rv=percent_pred_rv,
                rv_tlc=rv_tlc,
                lln_fev1_fvc=fev1fvc_res["FEV1 FVC LLN"],
                lln_tlc=tlc_res["tlc LLN"],
                lln_fev1=fev1_res["FEV1 LLN"],
                lln_fvc=fvc_res["FVC LLN"],
                lln_fef75=fef75_res["FEF75 LLN"],
                uln_rv_tlc=rvtlc_res["rvtlc ULN"],
            )
        except Exception:
            continue  # skip rows where calculation fails

        label = LABEL_MAP.get(diagnosis)
        if label is None:
            continue  # drop "Non-specific" and any unexpected values

        sex_enc = 0 if sex in ("male", "m") else 1

        records.append({
            "age":      age,
            "height":   height,
            "sex_enc":  sex_enc,
            "fev1":     fev1,
            "fvc":      fvc,
            "fev1_fvc": fev1_fvc_val,
            "label":    label,
        })

    return pd.DataFrame(records)


def _col_kwargs(run_config):
    """Extract column name kwargs from run_config."""
    return dict(
        col_age=run_config["col-age"],
        col_height=run_config["col-height"],
        col_sex=run_config["col-sex"],
        col_fev1=run_config["col-fev1"],
        col_fvc=run_config["col-fvc"],
        col_fev1_fvc=run_config["col-fev1-fvc"],
        col_fef75=run_config["col-fef75"],
        col_tlc=run_config["col-tlc"],
        col_rv=run_config["col-rv"],
        col_rv_tlc=run_config["col-rv-tlc"],
    )


def load_sim_data(partition_id, num_partitions, sim_data_path, run_config, test_fraction, seed):
    """Simulation data loader.

    Loads a single combined Excel file, preprocesses it, then slices an IID
    partition for this virtual client.
    """
    df_raw = pd.read_excel(sim_data_path)
    df = preprocess_pft_data(df_raw, **_col_kwargs(run_config))

    if len(df) == 0:
        raise ValueError(
            f"No valid rows after preprocessing simulation data at '{sim_data_path}'. "
            "Check column names and data values."
        )

    # IID partition: shuffle deterministically then slice
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    partition_size = len(df) // num_partitions
    start = partition_id * partition_size
    end = start + partition_size if partition_id < num_partitions - 1 else len(df)
    partition_df = df.iloc[start:end].reset_index(drop=True)

    train_df, valid_df, num_train, num_val = train_test_split(partition_df, test_fraction, seed)
    return transform_dataset_to_dmatrix(train_df), transform_dataset_to_dmatrix(valid_df), num_train, num_val


def load_local_data(data_path, run_config, test_fraction, seed):
    """Deployment data loader.

    Loads this client's own Excel file from data_path (set per-node in deployment).
    """
    df_raw = pd.read_excel(data_path)
    df = preprocess_pft_data(df_raw, **_col_kwargs(run_config))

    if len(df) == 0:
        raise ValueError(
            f"No valid rows after preprocessing local data at '{data_path}'. "
            "Check column names and data values."
        )

    train_df, valid_df, num_train, num_val = train_test_split(df, test_fraction, seed)
    return transform_dataset_to_dmatrix(train_df), transform_dataset_to_dmatrix(valid_df), num_train, num_val


def replace_keys(input_dict, match="-", target="_"):
    """Recursively replace match string with target string in dictionary keys."""
    new_dict = {}
    for key, value in input_dict.items():
        new_key = key.replace(match, target)
        if isinstance(value, dict):
            new_dict[new_key] = replace_keys(value, match, target)
        else:
            new_dict[new_key] = value
    return new_dict
