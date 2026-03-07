import os
import math
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path


class GLIReferenceValueCalculator:
    """
    The ReferenceValueCalculator is used for calculating the GLI2022 reference values for
    FEV1, FVC, FEV1FVC.
    
    Paper: https://doi.org/10.1164/rccm.202205-0963OC
    """

    def __init__(self):
        # Load the Excel file with the coefficients and the msplines
        dir = os.path.dirname(os.path.realpath(__file__))
        self.wb = load_workbook(dir + "/lookup_tables" + "/gli22_global_lookuptables.xlsx")

        # All the lung parameters for which the reference values can be calculated
        lung_params = ["FEV1", "FVC", "FEV1 FVC"]

        # Initialize all msplines and coefficients
        self.data = {}
        for lung_param in lung_params:
            ws_males = self.wb["Male " + lung_param]
            ws_females = self.wb["Female " + lung_param]

            self.data[lung_param] = {"males": {}, "females": {}}
            self.data[lung_param]["males"]["m_splines"] = [
                el[0].value for el in ws_males["B2":"B370"]
            ]
            self.data[lung_param]["males"]["s_splines"] = [
                el[0].value for el in ws_males["C2":"C370"]
            ]
            self.data[lung_param]["females"]["m_splines"] = [
                el[0].value for el in ws_females["B2":"B370"]
            ]
            self.data[lung_param]["females"]["s_splines"] = [
                el[0].value for el in ws_females["C2":"C370"]
            ]

        # Male FEV1 equations
        self.data["FEV1"]["males"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -11.399108
                + 2.462664 * math.log(height)
                - 0.011394 * math.log(age)
                + m_spline
            )
        )
        self.data["FEV1"]["males"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.256278 + 0.080729 * math.log(age) + s_spline
        )
        self.data["FEV1"]["males"]["L_equation"] = lambda age: 1.22703

        # Male FVC equations
        self.data["FVC"]["males"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -12.629131
                + 2.727421 * math.log(height)
                + 0.009174 * math.log(age)
                + m_spline
            )
        )
        self.data["FVC"]["males"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.195595 + 0.068466 * math.log(age) + s_spline
        )
        self.data["FVC"]["males"]["L_equation"] = lambda age: 0.9346

        # Male FEV1/FVC equations
        self.data["FEV1 FVC"]["males"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                1.022608
                - 0.218592 * math.log(height)
                - 0.027586 * math.log(age)
                + m_spline
            )
        )
        self.data["FEV1 FVC"]["males"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.882025 + 0.068889 * math.log(age) + s_spline
        )
        self.data["FEV1 FVC"]["males"]["L_equation"] = lambda age: (
            3.8243 - 0.3328 * math.log(age)
        )

        # Female FEV1 equations
        self.data["FEV1"]["females"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -10.901689
                + 2.385928 * math.log(height)
                - 0.076386 * math.log(age)
                + m_spline
            )
        )
        self.data["FEV1"]["females"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.364047 + 0.129402 * math.log(age) + s_spline
        )
        self.data["FEV1"]["females"]["L_equation"] = lambda age: 1.21388

        # Female FVC equations
        self.data["FVC"]["females"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -12.055901
                + 2.621579 * math.log(height)
                - 0.035975 * math.log(age)
                + m_spline
            )
        )
        self.data["FVC"]["females"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.310148 + 0.120428 * math.log(age) + s_spline
        )
        self.data["FVC"]["females"]["L_equation"] = lambda age: 0.899

        # Female FEV1/FVC equations
        self.data["FEV1 FVC"]["females"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                0.9189568
                - 0.1840671 * math.log(height)
                - 0.0461306 * math.log(age)
                + m_spline
            )
        )
        self.data["FEV1 FVC"]["females"]["S_equation"] = lambda age, s_spline: math.exp(
            -3.171582 + 0.144358 * math.log(age) + s_spline
        )
        self.data["FEV1 FVC"]["females"]["L_equation"] = lambda age: (
            6.6490 - 0.9920 * math.log(age)
        )

    def calc_lung_param(self, lung_param, sex, height, age):
        sex = str(sex)
        age = float(age)
        height = float(height)
        if sex.lower() in {"male", "m"}:
            sex = "male"
        else:
            sex = "female"

        m_splines = self.data[lung_param][sex + "s"]["m_splines"]
        s_splines = self.data[lung_param][sex + "s"]["s_splines"]

        # The array starts at age 3, the step is 0.25
        m_spline_index = max(0, min(len(m_splines) - 1, math.floor((age - 3) * 4)))
        s_spline_index = max(0, min(len(s_splines) - 1, math.floor((age - 3) * 4)))
        m_spline = m_splines[m_spline_index]
        s_spline = s_splines[s_spline_index]

        M_func = self.data[lung_param][sex + "s"]["M_equation"]
        S_func = self.data[lung_param][sex + "s"]["S_equation"]
        L_func = self.data[lung_param][sex + "s"]["L_equation"]

        M = M_func(height, age, m_spline)
        S = S_func(age, s_spline)
        L = L_func(age)

        lln = np.exp(np.log(1 - 1.645 * L * S) / L + np.log(M))
        uln = np.exp(np.log(1 + 1.645 * L * S) / L + np.log(M))
        predicted_value = M

        return {
            f"{lung_param} Predicted": predicted_value,
            f"{lung_param} LLN": lln,
            f"{lung_param} ULN": uln,
        }

    # Convenience Functions
    def calculate_fev1(self, sex, height, age):
        return self.calc_lung_param("FEV1", sex, height, age)

    def calculate_fvc(self, sex, height, age):
        return self.calc_lung_param("FVC", sex, height, age)

    def calculate_fev1fvc(self, sex, height, age):
        return self.calc_lung_param("FEV1 FVC", sex, height, age)


def use_calculator(file_path, age_col_name, height_col_name, sex_col_name):
    df = pd.read_excel(file_path)
    dir = os.path.dirname(file_path)
    file_name = Path(file_path).stem

    # Check for invalid values
    print(f"Age range: {df[age_col_name].min()} - {df[age_col_name].max()}")
    print(f"Height range: {df[height_col_name].min()} - {df[height_col_name].max()}")
    print(f"Missing ages: {df[age_col_name].isna().sum()}")
    print(f"Missing heights: {df[height_col_name].isna().sum()}")

    # new columns first
    new_cols = [
        "FEV1 Predicted",
        "FEV1 LLN",
        "FEV1 ULN",
        "FVC Predicted",
        "FVC LLN",
        "FVC ULN",
        "FEV1 FVC Predicted",
        "FEV1 FVC LLN",
        "FEV1 FVC ULN",
    ]
    for col in new_cols:
        df[col] = np.nan

    calc = GLIReferenceValueCalculator()
    for idx, row in df.iterrows():
        sex = row[sex_col_name]
        height = row[height_col_name]
        age = row[age_col_name]

        if pd.isna(age) or pd.isna(height) or pd.isna(sex) or height < 0 or age < 0:
            continue
        fev1_results = calc.calculate_fev1(sex, height, age)
        fvc_results = calc.calculate_fvc(sex, height, age)
        fev1fvc_results = calc.calculate_fev1fvc(sex, height, age)

        # Assign values using df.at[index, column]
        for key, value in fev1_results.items():
            df.at[idx, key] = value
        for key, value in fvc_results.items():
            df.at[idx, key] = value
        for key, value in fev1fvc_results.items():
            df.at[idx, key] = value

    df.to_excel(dir + "/" + file_name + "_GLI.xlsx", index=False)
    return dir + "/" + file_name + "_GLI.xlsx"


def test_reference_value_calculator():
    age = 80
    height = 160
    sex = "female"
    calc = GLIReferenceValueCalculator()
    print(f"\n sex: {sex}, age: {age}, height:{height} \n")
    print("FEV1 \n", calc.calculate_fev1(sex, height, age))
    print("FVC \n", calc.calculate_fvc(sex, height, age))
    print("FEV1 FVC \n", calc.calculate_fev1fvc(sex, height, age))


if __name__ == "__main__":
    file_path = input("File Path: ")
    file_name = Path(file_path).stem
    age_col_name = input("Input the verbatim column name for age: ")
    height_col_name = input("Input the verbatim column name for height: ")
    sex_col_name = input("Input the verbatim column name for sex: ")
    success_file_path = use_calculator(file_path, age_col_name, height_col_name, sex_col_name)
    print(f"Successfully processed {file_name}, the new file with added columns is stored at {success_file_path}")
