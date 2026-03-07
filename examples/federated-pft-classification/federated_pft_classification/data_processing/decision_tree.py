import os
import pandas as pd

def interpret_pft(fev1_fvc, tlc, fev1, fvc, fef75, percent_pred_fvc,
                  percent_pred_rv, rv_tlc,
                  lln_fev1_fvc, lln_tlc, lln_fev1, lln_fvc, lln_fef75, uln_rv_tlc):
    """
    Interpret pulmonary function test results based on the decision tree.

    Parameters:
        fev1_fvc      : measured FEV1/FVC ratio
        tlc           : measured TLC
        fev1          : measured FEV1
        fvc           : measured FVC
        fef75         : measured FEF75
        percent_pred_fvc : %predicted FVC
        percent_pred_rv  : %predicted RV
        rv_tlc        : measured RV/TLC ratio
        lln_fev1_fvc  : lower limit of normal for FEV1/FVC
        lln_tlc       : lower limit of normal for TLC
        lln_fev1      : lower limit of normal for FEV1
        lln_fvc       : lower limit of normal for FVC
        lln_fef75     : lower limit of normal for FEF75
        uln_rv_tlc    : upper limit of normal for RV/TLC

    Returns:
        str: diagnosis code
             R+AO  = mixed restriction + obstruction
             AO    = obstruction
             R+SAO = restriction + small airway obstruction
             SAO   = small airway obstruction
             R     = restriction
             GT    = gas trapping
             N     = normal
             Non-specific
    """

    if fev1_fvc < lln_fev1_fvc:
        # --- Obstructive branch (FEV1/FVC < LLN) ---
        if tlc < lln_tlc:
            return "R+AO"   # mixed restriction + obstruction
        else:
            return "AO"     # obstruction

    else:
        # --- FEV1/FVC >= LLN ---
        if fvc < lln_fvc and fev1 < lln_fev1:
            # Left sub-branch: FEF75 < 0.25*FVC?
            if fef75 < 0.25 * fvc:
                if tlc < lln_tlc:
                    return "R+SAO"
                else:
                    return "SAO"
            else:
                if tlc < lln_tlc:
                    return "R"
                else:
                    if percent_pred_rv > 150 or rv_tlc > uln_rv_tlc:
                        return "GT"
                    else:
                        return "Non-specific"

        else:
            # Right sub-branch: 90 <= %pred FVC <= 110?
            if 90 <= percent_pred_fvc <= 110:
                # FEF75 < LLN?
                if fef75 < lln_fef75:
                    if tlc < lln_tlc:
                        return "R+SAO"
                    else:
                        return "SAO"
                else:
                    if tlc < lln_tlc:
                        return "R"
                    else:
                        if percent_pred_rv > 150 or rv_tlc > uln_rv_tlc:
                            return "GT"
                        else:
                            return "N"
            else:
                # %pred FVC outside 90–110
                # FEF75 < 0.25*FVC?
                if fef75 < 0.25 * fvc:
                    if tlc < lln_tlc:
                        return "R+SAO"
                    else:
                        return "SAO"
                else:
                    if tlc < lln_tlc:
                        return "R"
                    else:
                        if percent_pred_rv > 150 or rv_tlc > uln_rv_tlc:
                            return "GT"
                        else:
                            return "N"

            
PARAMETERS = [
    ("fev1_fvc",       "FEV1/FVC ratio (measured)"),
    ("tlc",            "TLC (measured)"),
    ("fev1",           "FEV1 (measured)"),
    ("fvc",            "FVC (measured)"),
    ("fef75",          "FEF75 (measured)"),
    ("rv_tlc",         "RV/TLC ratio (measured)"),
    ("lln_fev1_fvc",   "Lower limit of normal for FEV1/FVC"),
    ("lln_tlc",        "Lower limit of normal for TLC"),
    ("lln_fev1",       "Lower limit of normal for FEV1"),
    ("lln_fvc",        "Lower limit of normal for FVC"),
    ("lln_fef75",      "Lower limit of normal for FEF75"),
    ("uln_rv_tlc",     "Upper limit of normal for RV/TLC"),
]

# These pairs are used to compute %predicted = (actual / predicted_normal) * 100
DERIVED_PARAMETERS = [
    ("fvc_pred_normal", "FVC predicted normal value [used to compute %pred FVC]"),
    ("rv_actual",       "RV actual measured value   [used to compute %pred RV]"),
    ("rv_pred_normal",  "RV predicted normal value  [used to compute %pred RV]"),
]

def get_column_mapping(df_columns):
    """Prompt the user to map each decision-tree parameter to a column in the Excel file."""
    named_columns = [c for c in df_columns if str(c).strip() and not str(c).startswith("Unnamed:")]
    print("\nAvailable columns in the Excel file:")
    for i, col in enumerate(named_columns):
        print(f"  [{i}] {col}")

    print("\nFor each parameter, enter the exact column name (or its index from the list above).")
    mapping = {}
    for param, description in PARAMETERS + DERIVED_PARAMETERS:
        while True:
            raw = input(f"  {description} [{param}]: ").strip()
            # Allow entry by index
            if raw.isdigit() and int(raw) < len(named_columns):
                col = named_columns[int(raw)]
            else:
                col = raw
            if col in named_columns:
                mapping[param] = col
                break
            else:
                print(f"    Column '{col}' not found. Please try again.")
    return mapping


def run_on_excel(excel_path):
    """
    Load an Excel file, prompt for column mappings, run interpret_pft on every row,
    and save the results to decision_tree_predicted/<original_filename>.
    """
    df = pd.read_excel(excel_path)
    mapping = get_column_mapping(list(df.columns))

    results = []
    for _, row in df.iterrows():
        try:
            percent_pred_fvc = row[mapping["fvc"]] / row[mapping["fvc_pred_normal"]] * 100
            percent_pred_rv  = row[mapping["rv_actual"]]  / row[mapping["rv_pred_normal"]]  * 100
            result = interpret_pft(
                fev1_fvc=row[mapping["fev1_fvc"]],
                tlc=row[mapping["tlc"]],
                fev1=row[mapping["fev1"]],
                fvc=row[mapping["fvc"]],
                fef75=row[mapping["fef75"]],
                percent_pred_fvc=percent_pred_fvc,
                percent_pred_rv=percent_pred_rv,
                rv_tlc=row[mapping["rv_tlc"]],
                lln_fev1_fvc=row[mapping["lln_fev1_fvc"]],
                lln_tlc=row[mapping["lln_tlc"]],
                lln_fev1=row[mapping["lln_fev1"]],
                lln_fvc=row[mapping["lln_fvc"]],
                lln_fef75=row[mapping["lln_fef75"]],
                uln_rv_tlc=row[mapping["uln_rv_tlc"]],
            )
        except Exception as e:
            result = f"ERROR: {e}"
        results.append(result)

    df["decision_tree_prediction"] = results

    out_dir = os.path.join(os.path.dirname(os.path.abspath(excel_path)), "decision_tree_predicted")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(excel_path))
    df.to_excel(out_path, index=False)
    print(f"\nDone. Output saved to: {out_path}")


if __name__ == "__main__":
    path = input("Enter the path to the Excel file: ").strip().strip("'\"")
    run_on_excel(path)