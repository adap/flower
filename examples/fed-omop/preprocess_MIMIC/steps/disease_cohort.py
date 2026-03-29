
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

def read_icd_mapping(map_path: str) -> pd.DataFrame:
    """Reads in mapping table for converting ICD9 to ICD10 codes"""

    mapping = pd.read_csv(map_path, header=0, delimiter="\t")
    mapping.diagnosis_description = mapping.diagnosis_description.apply(str.lower)
    return mapping


def get_diagnosis_icd(module_path: str) -> pd.DataFrame:
    """Reads in diagnosis_icd table"""

    return pd.read_csv(
        module_path + "/hosp/diagnoses_icd.csv.gz", compression="gzip", header=0
    )


def standardize_icd(
    mapping: pd.DataFrame, diag: pd.DataFrame, map_code_col="diagnosis_code", root=True
) -> str:
    """Takes an ICD9 -> ICD10 mapping table and a diagnosis dataframe;
    adds column with converted ICD10 column"""

    count = 0
    code_cols = mapping.columns
    errors = []

    def icd_9to10(icd):
        """Function use to apply over the diag DataFrame for ICD9->ICD10 conversion"""
        # If root is true, only map an ICD 9 -> 10 according to the
        # ICD9's root (first 3 digits)
        if root:
            icd = icd[:3]

        if map_code_col not in code_cols:
            errors.append(f"ICD NOT FOUND: {icd}")
            return np.nan

        matches = mapping.loc[mapping[map_code_col] == icd]
        if matches.shape[0] == 0:
            errors.append(f"ICD NOT FOUND: {icd}")
            return np.nan

        return mapping.loc[mapping[map_code_col] == icd].icd10cm.iloc[0]

    # Create new column with original codes as default
    col_name = "root_icd10_convert"
    diag[col_name] = diag["icd_code"].values

    # Group identical ICD9 codes, then convert all ICD9 codes within
    # a group to ICD10
    for code, group in diag.loc[diag.icd_version == 9].groupby(by="icd_code"):
        new_code = icd_9to10(code)
        for idx in group.index.values:
            # Modify values of original df at the indexes in the groups
            diag.at[idx, col_name] = new_code

        count += group.shape[0]
        #print(f"{count}/{diag.shape[0]} rows processed")

    # Column for just the roots of the converted ICD10 column
    diag["root"] = diag[col_name].apply(lambda x: x[:3] if type(x) is str else np.nan)



def preproc_icd_module(h_ids,
    module_path: str, ICD10_code: str, icd_map_path: str
) -> tuple:
    """Takes an module dataset with ICD codes and puts it in long_format,
    mapping ICD-codes by a mapping table path"""

    diag = get_diagnosis_icd(module_path)
    icd_map = read_icd_mapping(icd_map_path)

    standardize_icd(icd_map, diag, root=True)

    # patient ids that have at least 1 record of the given ICD10 code category
    diag.dropna(subset=["root"], inplace=True)
    pos_ids = pd.DataFrame(
        diag.loc[diag.root.str.contains(ICD10_code)].hadm_id.unique(),
        columns=["hadm_id"]
    )
    return pos_ids


def extract_diag_cohort(
    h_ids,
    label: str,
    module_path,
    icd_map_path="./utils/mappings/ICD9_to_ICD10_mapping.txt"
) -> str:
    """Takes UserInterface parameters, then creates and saves a labelled cohort
    summary, and error file"""

    cohort = preproc_icd_module(h_ids,
        module_path, label, icd_map_path
    )

    return cohort


