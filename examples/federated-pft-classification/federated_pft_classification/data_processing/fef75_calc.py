from openpyxl import load_workbook
import math
import numpy as np
import os


class FEF75ValueCalculator:
    """
    The FEF75ValueCalculator is used for calculating the GLI2012 reference values for
    FEF75.
    """

    def __init__(self):
        # Load the Excel file with the coefficients and the msplines
        dir = os.path.dirname(os.path.realpath(__file__))
        self.wb = load_workbook(dir + "/lookup_tables" + "/fef75_lookuptable.xlsx")

        # All the lung parameters for which the reference values can be calculated
        lung_params = ["FEF75"]

        # Initialize all msplines and coefficients
        self.data = {}
        for lung_param in lung_params:
            ws_males = self.wb[lung_param + " males"]
            ws_females = self.wb[lung_param + " females"]

            self.data[lung_param] = {"males": {}, "females": {}}
            self.data[lung_param]["males"]["m_splines"] = [
                el[0].value for el in ws_males["D3":"D351"]
            ]
            self.data[lung_param]["males"]["m_coeffs"] = [
                el[0].value for el in ws_males["I4":"I10"]
            ]
            self.data[lung_param]["males"]["s_splines"] = [
                el[0].value for el in ws_males["E3":"E351"]
            ]
            self.data[lung_param]["males"]["s_coeffs"] = [
                el[0].value for el in ws_males["L4":"L10"]
            ]
            self.data[lung_param]["males"]["l_coeffs"] = [
                el[0].value for el in ws_males["O4":"O6"]
            ]

            self.data[lung_param]["females"]["m_splines"] = [
                el[0].value for el in ws_females["D3":"D351"]
            ]
            self.data[lung_param]["females"]["m_coeffs"] = [
                el[0].value for el in ws_females["I4":"I10"]
            ]
            self.data[lung_param]["females"]["s_splines"] = [
                el[0].value for el in ws_females["E3":"E351"]
            ]
            self.data[lung_param]["females"]["s_coeffs"] = [
                el[0].value for el in ws_females["L4":"L10"]
            ]
            self.data[lung_param]["females"]["l_coeffs"] = [
                el[0].value for el in ws_females["O4":"O6"]
            ]

    def calc_lung_param(self, lung_param, sex, height, age, race):
        """The base function or LLN doing the calculation work with the splines and the coefficients."""
        sex = str(sex)
        age = float(age)
        height = float(height)
        if sex.lower() in {"male", "m"}:
            sex = "male"
        else:
            sex = "female"

        m_splines = self.data[lung_param][sex + "s"]["m_splines"]
        m_coeffs = self.data[lung_param][sex + "s"]["m_coeffs"]
        s_splines = self.data[lung_param][sex + "s"]["s_splines"]
        s_coeffs = self.data[lung_param][sex + "s"]["s_coeffs"]
        l_coeffs = self.data[lung_param][sex + "s"]["l_coeffs"]

        # The array starts at age 3, the step is 0.25
        m_spline_index = max(0, min(len(m_splines) - 1, math.floor((age - 3) * 4)))
        s_spline_index = max(0, min(len(s_splines) - 1, math.floor((age - 3) * 4)))
        m_spline = m_splines[m_spline_index]
        s_spline = s_splines[s_spline_index]
        M = np.exp(
            m_coeffs[0]
            + m_coeffs[1] * np.log(height)
            + m_coeffs[2] * np.log(age)
            + m_coeffs[3] * (race == "AfrAm")
            + m_coeffs[4] * (race == "NEAsia")
            + m_coeffs[5] * (race == "SEAsia")
            + m_coeffs[6] * (race == "other")
            + m_spline
        )
        S = np.exp(
            s_coeffs[0]
            + s_coeffs[2] * np.log(age)
            + s_coeffs[3] * (race == "AfrAm")
            + s_coeffs[4] * (race == "NEAsia")
            + s_coeffs[5] * (race == "SEAsia")
            + s_coeffs[6] * (race == "other")
            + s_spline
        )
        L = l_coeffs[0] + l_coeffs[2] * np.log(age)

        uln = np.exp(np.log(1 + 1.645 * L * S) / L + np.log(M))
        lln = np.exp(np.log(1 - 1.645 * L * S) / L + np.log(M))
        predicted_value = M

        return {
            f"{lung_param} Predicted": predicted_value,
            f"{lung_param} LLN": lln,
            f"{lung_param} ULN": uln,
        }

    def calculate_fef75(self, sex, height, age, race="Cau"):
        return self.calc_lung_param("FEF75", sex, height, age, race)
