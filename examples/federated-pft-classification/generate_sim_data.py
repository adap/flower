"""
Generate a synthetic simulation dataset (data/simulation_data.xlsx) with
patients representative of all 7 lung-function categories.

Values are set relative to the GLI/ERS LLN values for each patient's
biometrics so that the decision tree in the training pipeline produces the
intended labels.
"""

import os
import sys

import numpy as np
import pandas as pd

# Make sure the package root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from federated_pft_classification.data_processing.gli22_calc import GLIReferenceValueCalculator
from federated_pft_classification.data_processing.ers21_lung_volumes_calc import ERSLungVolumeCalculator
from federated_pft_classification.data_processing.fef75_calc import FEF75ValueCalculator

gli22  = GLIReferenceValueCalculator()
ers21  = ERSLungVolumeCalculator()
fef75c = FEF75ValueCalculator()

rng = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helper: random biometrics + reference values
# ---------------------------------------------------------------------------
def biometrics():
    sex = rng.choice(["male", "female"])
    age = float(rng.uniform(25, 75))
    height = float(rng.uniform(160, 190) if sex == "male" else rng.uniform(150, 175))
    return sex, age, height


def ref(sex, height, age):
    tlc_r    = ers21.calculate_tlc(sex, height, age)
    rv_r     = ers21.calculate_rv(sex, height, age)
    rvtlc_r  = ers21.calculate_rvtlc(sex, height, age)
    fev1_r   = gli22.calculate_fev1(sex, height, age)
    fvc_r    = gli22.calculate_fvc(sex, height, age)
    fev1fvc_r = gli22.calculate_fev1fvc(sex, height, age)
    fef75_r  = fef75c.calculate_fef75(sex, height, age)
    return {
        "tlc_pred":   tlc_r["tlc Predicted"],
        "tlc_lln":    tlc_r["tlc LLN"],
        "rv_pred":    rv_r["rv Predicted"],
        "rvtlc_uln":  rvtlc_r["rvtlc ULN"],
        "fev1_pred":  fev1_r["FEV1 Predicted"],
        "fev1_lln":   fev1_r["FEV1 LLN"],
        "fvc_pred":   fvc_r["FVC Predicted"],
        "fvc_lln":    fvc_r["FVC LLN"],
        "fev1fvc_lln": fev1fvc_r["FEV1 FVC LLN"],
        "fef75_lln":  fef75_r["FEF75 LLN"],
    }


def j(v, pct=0.05):
    """Small random jitter (±pct) so values aren't identical."""
    return float(v * (1 + rng.uniform(-pct, pct)))


# ---------------------------------------------------------------------------
# Category generators (returns a dict row or None on failure)
# ---------------------------------------------------------------------------
def make_normal():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fvc     = j(r["fvc_pred"])                          # ~100 %pred → right branch
    fev1fvc = j(r["fev1fvc_lln"] * 1.15)               # > LLN
    fev1    = j(r["fev1_pred"])
    fef75   = j(r["fef75_lln"] * 1.5)                   # > LLN_fef75
    tlc     = j(r["tlc_pred"])                           # ≥ LLN
    rv      = j(r["rv_pred"])                            # ~100 %pred (not elevated)
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


def make_ao():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fev1fvc = j(r["fev1fvc_lln"] * 0.85)               # < LLN  → obstructive branch
    fvc     = j(r["fvc_pred"])
    fev1    = fvc * fev1fvc
    fef75   = j(r["fef75_lln"] * 0.5)
    tlc     = j(r["tlc_pred"] * 1.05)                   # ≥ LLN → pure AO (not mixed)
    rv      = j(r["rv_pred"] * 1.3)
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


def make_r_ao():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fev1fvc = j(r["fev1fvc_lln"] * 0.85)               # < LLN
    fvc     = j(r["fvc_pred"] * 0.75)
    fev1    = fvc * fev1fvc
    fef75   = j(r["fef75_lln"] * 0.5)
    tlc     = j(r["tlc_lln"] * 0.88)                   # < LLN → mixed
    rv      = j(r["rv_pred"] * 0.8)
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


def make_restriction():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fev1fvc = j(r["fev1fvc_lln"] * 1.1)                # ≥ LLN
    fvc     = j(r["fvc_lln"] * 0.88)                   # < LLN → left branch condition
    fev1    = j(r["fev1_lln"] * 0.88)                  # < LLN → left branch condition
    fef75   = j(fvc * 0.35)                             # ≥ 0.25*fvc (not small airway)
    tlc     = j(r["tlc_lln"] * 0.88)                   # < LLN → R
    rv      = j(r["rv_pred"] * 0.8)
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


def make_gas_trapping():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fvc     = j(r["fvc_pred"])                          # ~100 %pred → right branch
    fev1fvc = j(r["fev1fvc_lln"] * 1.1)                # ≥ LLN
    fev1    = j(r["fev1_pred"])
    fef75   = j(r["fef75_lln"] * 1.3)                   # ≥ LLN_fef75
    tlc     = j(r["tlc_pred"] * 1.05)                   # ≥ LLN
    rv      = j(r["rv_pred"] * 1.6)                     # > 150 %pred → gas trapping
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


def make_sao():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fvc     = j(r["fvc_pred"])                          # ~100 %pred → right branch
    fev1fvc = j(r["fev1fvc_lln"] * 1.1)                # ≥ LLN
    fev1    = j(r["fev1_pred"])
    fef75   = j(r["fef75_lln"] * 0.65)                  # < LLN → small airway
    tlc     = j(r["tlc_pred"])                           # ≥ LLN → SAO (not R+SAO)
    rv      = j(r["rv_pred"] * 1.1)
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


def make_r_sao():
    sex, age, height = biometrics()
    r = ref(sex, height, age)
    fvc     = j(r["fvc_pred"])                          # ~100 %pred → right branch
    fev1fvc = j(r["fev1fvc_lln"] * 1.1)                # ≥ LLN
    fev1    = j(r["fev1_pred"])
    fef75   = j(r["fef75_lln"] * 0.65)                  # < LLN → small airway
    tlc     = j(r["tlc_lln"] * 0.88)                   # < LLN → R+SAO
    rv      = j(r["rv_pred"] * 0.8)
    rv_tlc  = rv / tlc
    return dict(age=age, sex=sex, height=height,
                fev1=fev1, fvc=fvc, fev1_fvc=fev1fvc,
                fef75=fef75, tlc=tlc, rv=rv, rv_tlc=rv_tlc)


# ---------------------------------------------------------------------------
# Generate dataset
# ---------------------------------------------------------------------------
GENERATORS = [
    (make_normal,      "N",     50),
    (make_ao,          "AO",    50),
    (make_r_ao,        "R+AO",  30),
    (make_restriction, "R",     40),
    (make_gas_trapping,"GT",    30),
    (make_sao,         "SAO",   40),
    (make_r_sao,       "R+SAO", 30),
]

rows = []
counts = {}
for fn, label, n in GENERATORS:
    for _ in range(n):
        try:
            row = fn()
            row["expected_label"] = label   # for human inspection only (not used in training)
            rows.append(row)
        except Exception as e:
            print(f"  Warning: {label} row failed: {e}")
    counts[label] = n

df = pd.DataFrame(rows)
# Drop the expected_label column before saving (training uses only PFT measurements)
df_out = df.drop(columns=["expected_label"])

os.makedirs("data", exist_ok=True)
df_out.to_excel("data/simulation_data.xlsx", index=False)

print(f"\nGenerated {len(df_out)} patients → data/simulation_data.xlsx")
print("Target distribution:")
for label, n in counts.items():
    print(f"  {label:6s}: {n}")
print("\nColumn preview:")
print(df_out.head(3).to_string())
