import os
import math
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from pathlib import Path


class ERSLungVolumeCalculator:
    """
    The ERSLungVolumeCalculator is used for calculating the ERS2021 reference values for
    Lung Volume Parameters: tlc, rv, rv/tlc.
    """

    def __init__(self):
        # Load the Excel file with the coefficients and the msplines
        dir = os.path.dirname(os.path.realpath(__file__))
        self.wb = load_workbook(dir + "/lookup_tables" + "/ers21_lung_volume_lookuptables.xlsx")

        # All the lung parameters for which the reference values can be calculated
        lung_params = ["tlc", "rv", "rvtlc"]

        # Initialize all msplines and coefficients
        self.data = {}
        for lung_param in lung_params:
            ws_males = self.wb[lung_param + "_m_lookuptable"]
            ws_females = self.wb[lung_param + "_f_lookuptable"]

            self.data[lung_param] = {"males": {}, "females": {}}
            self.data[lung_param]["males"]["m_splines"] = [
                el[0].value for el in ws_males["B2":"B302"]
            ]
            self.data[lung_param]["males"]["s_splines"] = [
                el[0].value for el in ws_males["C2":"C302"]
            ]
            self.data[lung_param]["females"]["m_splines"] = [
                el[0].value for el in ws_females["B2":"B302"]
            ]
            self.data[lung_param]["females"]["s_splines"] = [
                el[0].value for el in ws_females["C2":"C302"]
            ]

        # TLC equations
        self.data["tlc"]["males"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -10.5861 + 0.1433 * math.log(age) + 2.3155 * math.log(height) + m_spline
            )
        )
        self.data["tlc"]["males"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.0616143 - 0.0008534 * age + s_spline
        )
        self.data["tlc"]["males"]["L_equation"] = lambda: 0.9337

        self.data["tlc"]["females"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -10.1128 + 0.1062 * math.log(age) + 2.2259 * math.log(height) + m_spline
            )
        )
        self.data["tlc"]["females"]["S_equation"] = lambda age, s_spline: math.exp(
            -2.0999321 + 0.0001564 * age + s_spline
        )
        self.data["tlc"]["females"]["L_equation"] = lambda: 0.4636

        # RV equations
        self.data["rv"]["males"]["M_equation"] = lambda height, age, m_spline: math.exp(
            -2.37211 + 0.01346 * age + 0.01307 * height + m_spline
        )
        self.data["rv"]["males"]["S_equation"] = lambda age, s_spline: math.exp(
            -0.878572 - 0.007032 * age + s_spline
        )
        self.data["rv"]["males"]["L_equation"] = lambda: 0.5931

        self.data["rv"]["females"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                -2.50593 + 0.01307 * age + 0.01379 * height + m_spline
            )
        )
        self.data["rv"]["females"]["S_equation"] = lambda age, s_spline: math.exp(
            -0.902550 - 0.006005 * age + s_spline
        )
        self.data["rv"]["females"]["L_equation"] = lambda: 0.4197

        # RV/TLC equations
        self.data["rvtlc"]["males"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                2.634 + 0.01302 * age - 0.00008862 * height + m_spline
            )
        )
        self.data["rvtlc"]["males"]["S_equation"] = lambda age, s_spline: math.exp(
            -0.96804 - 0.01004 * age + s_spline
        )
        self.data["rvtlc"]["males"]["L_equation"] = lambda: 0.8646

        self.data["rvtlc"]["females"]["M_equation"] = (
            lambda height, age, m_spline: math.exp(
                2.666 + 0.01411 * age - 0.00003689 * height + m_spline
            )
        )
        self.data["rvtlc"]["females"]["S_equation"] = lambda age, s_spline: math.exp(
            -0.976602 - 0.009679 * age + s_spline
        )
        self.data["rvtlc"]["females"]["L_equation"] = lambda: 0.8037

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

        # The array starts at age 5, the step is 0.25, so multiply 4 to get the right index
        m_spline_index = max(0, min(len(m_splines) - 1, math.floor((age - 5) * 4)))
        s_spline_index = max(0, min(len(s_splines) - 1, math.floor((age - 5) * 4)))
        m_spline = m_splines[m_spline_index]
        s_spline = s_splines[s_spline_index]

        M_func = self.data[lung_param][sex + "s"]["M_equation"]
        S_func = self.data[lung_param][sex + "s"]["S_equation"]
        L_func = self.data[lung_param][sex + "s"]["L_equation"]

        M = M_func(height, age, m_spline)
        S = S_func(age, s_spline)
        L = L_func()

        lln = np.exp(np.log(1 - 1.645 * L * S) / L + np.log(M))
        uln = np.exp(np.log(1 + 1.645 * L * S) / L + np.log(M))
        predicted_value = M

        return {
            f"{lung_param} Predicted": predicted_value,
            f"{lung_param} LLN": lln,
            f"{lung_param} ULN": uln,
        }

    # Convenience Functions
    def calculate_tlc(self, sex, height, age):
        return self.calc_lung_param("tlc", sex, height, age)

    def calculate_rv(self, sex, height, age):
        return self.calc_lung_param("rv", sex, height, age)

    def calculate_rvtlc(self, sex, height, age):
        return self.calc_lung_param("rvtlc", sex, height, age)


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
        "tlc Predicted",
        "tlc LLN",
        "tlc ULN",
        "rv Predicted",
        "rv LLN",
        "rv ULN",
        "rvtlc Predicted",
        "rvtlc LLN",
        "rvtlc ULN",
    ]
    for col in new_cols:
        df[col] = np.nan

    calc = ERSLungVolumeCalculator()
    for idx, row in df.iterrows():
        sex = row[sex_col_name]
        height = row[height_col_name]
        age = row[age_col_name]

        if pd.isna(age) or pd.isna(height) or pd.isna(sex) or height < 0 or age < 0:
            continue
        tlc_results = calc.calculate_tlc(sex, height, age)
        rv_results = calc.calculate_rv(sex, height, age)
        rvtlc_results = calc.calculate_rvtlc(sex, height, age)

        # Assign values using df.at[index, column]
        for key, value in tlc_results.items():
            df.at[idx, key] = value
        for key, value in rv_results.items():
            df.at[idx, key] = value
        for key, value in rvtlc_results.items():
            df.at[idx, key] = value

    df.to_excel(dir + "/" + file_name + "_ERS_vol.xlsx", index=False)
    return dir + "/" + file_name + "_ERS_vol.xlsx"


def test_reference_value_calculator():
    age = 80
    height = 160
    sex = "female"
    calc = ERSLungVolumeCalculator()
    print(f"\n sex: {sex}, age: {age}, height:{height} \n")
    print("tlc \n", calc.calculate_tlc(sex, height, age))
    print("rv \n", calc.calculate_rv(sex, height, age))
    print("rvtlc \n", calc.calculate_rvtlc(sex, height, age))


if __name__ == "__main__":
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
