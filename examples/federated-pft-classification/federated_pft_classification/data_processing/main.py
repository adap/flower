import os
import math
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path

from .gli22_calc import GLIReferenceValueCalculator
from .ers21_lung_volumes_calc import ERSLungVolumeCalculator
from .fef75_calc import FEF75ValueCalculator


def use_calculator(file_path, age_col_name, height_col_name, sex_col_name):
    df = pd.read_excel(file_path)
    dir = os.path.dirname(file_path)
    file_name = Path(file_path).stem

    # Check for invalid values
    print(f"Age range: {df[age_col_name].min()} - {df[age_col_name].max()}")
    print(f"Height range: {df[height_col_name].min()} - {df[height_col_name].max()}")
    print(f"Missing ages: {df[age_col_name].isna().sum()}")
    print(f"Missing heights: {df[height_col_name].isna().sum()}")

    #####################################
    # new columns for lung volume metrics
    #####################################

    new_cols = [
        "tlc Predicted",
        "tlc LLN",
        "tlc ULN",
        "rv Predicted",
        "rv LLN",
        "rv ULN",
        "rvtlc Predicted",
        "rvtlc LLN",
        "rvtlc ULN",
        "FEV1 Predicted",
        "FEV1 LLN",
        "FEV1 ULN",
        "FVC Predicted",
        "FVC LLN",
        "FVC ULN",
        "FEV1 FVC Predicted",
        "FEV1 FVC LLN",
        "FEV1 FVC ULN",
        "FEF75 Predicted",
        "FEF75 LLN",
        "FEF75 ULN",
        "",
    ]
    for col in new_cols:
        df[col] = np.nan

    ERScalc = ERSLungVolumeCalculator()
    GLI22calc = GLIReferenceValueCalculator()
    FEF75calc = FEF75ValueCalculator()

    for idx, row in df.iterrows():
        sex = row[sex_col_name]
        height = row[height_col_name]
        age = row[age_col_name]

        if pd.isna(age) or pd.isna(height) or pd.isna(sex) or height < 0 or age < 0:
            continue

        tlc_results = ERScalc.calculate_tlc(sex, height, age)
        rv_results = ERScalc.calculate_rv(sex, height, age)
        rvtlc_results = ERScalc.calculate_rvtlc(sex, height, age)
        fev1_results = GLI22calc.calculate_fev1(sex, height, age)
        fvc_results = GLI22calc.calculate_fvc(sex, height, age)
        fev1fvc_results = GLI22calc.calculate_fev1fvc(sex, height, age)
        fef75_results = FEF75calc.calculate_fef75(sex, height, age)

        # Assign values using df.at[index, column]
        for key, value in tlc_results.items():
            df.at[idx, key] = value
        for key, value in rv_results.items():
            df.at[idx, key] = value
        for key, value in rvtlc_results.items():
            df.at[idx, key] = value
        for key, value in fev1_results.items():
            df.at[idx, key] = value
        for key, value in fvc_results.items():
            df.at[idx, key] = value
        for key, value in fev1fvc_results.items():
            df.at[idx, key] = value
        for key, value in fef75_results.items():
            df.at[idx, key] = value

    output_dir = os.path.join(dir, "ref_values_append")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{file_name}_ref_values.xlsx")
    df.to_excel(output_path, index=False)
    return output_path


def test_reference_value_calculator():
    age = 60
    height = 180
    sex = "female"
    ERScalc = ERSLungVolumeCalculator()
    GLI22calc = GLIReferenceValueCalculator()
    FEF75calc = FEF75ValueCalculator()
    print(f"\n sex: {sex}, age: {age}, height:{height} \n")
    print("tlc \n", ERScalc.calculate_tlc(sex, height, age))
    print("rv \n", ERScalc.calculate_rv(sex, height, age))
    print("rvtlc \n", ERScalc.calculate_rvtlc(sex, height, age))
    print("FEV1 \n", GLI22calc.calculate_fev1(sex, height, age))
    print("FVC \n", GLI22calc.calculate_fvc(sex, height, age))
    print("FEV1 FVC \n", GLI22calc.calculate_fev1fvc(sex, height, age))
    print("FEF75 \n", FEF75calc.calculate_fef75(sex, height, age))


if __name__ == "__main__":
    # test_reference_value_calculator()

    file_path = input("File Path: ")
    file_name = Path(file_path).stem
    age_col_name = input("Input the verbatim column name for age: ")
    height_col_name = input("Input the verbatim column name for height: ")
    sex_col_name = input("Input the verbatim column name for sex: ")
    success_file_path = use_calculator(
        file_path, age_col_name, height_col_name, sex_col_name
    )
    print(
        f"Successfully processed {file_name}, the new file with added columns is stored at {success_file_path}"
    )
