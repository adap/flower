
import json
from dataclasses import asdict, dataclass
from os.path import splitext

class SavableConfig:
    """A generic Config dataclass that can be saved to a json format"""

    def save_to_json(self, path: str) -> None:
        """Save this instance as json

        Parameters
        ----------
        path : str
            The .json filepath, e.g. 'directory/config.json'
        """
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=4)
    
    def load_from_json(self, path: str) -> None:
        """Load this instance from a json file

        Parameters
        ----------
        path : str
            The .json filepath, e.g. 'directory/config.json'
        """
        with open(path, "r") as f:
            data = json.load(f)
            for key, value in data.items():
                setattr(self, key, value)

@dataclass
class PrepocessConfig(SavableConfig):
    """All parameters for the preprocess step
    
    Parameters
    ----------

    """


    Version: str = "2.2"
    # Paths
    RawDataPath: str = "data/mimic-IV/2.2"
    # Preprocessing parameters
    Task: str = "Mortality"  # mortality, LengthOfStay, readmission , phenotype 
    # task specific parameters 
    Mortality_prediction_horizon: int = 2  # in hours, e.g. 2 for predicting mortality within next 24 hours
    LengthOfStay_greater_or_equal_threshold: int = 3  # in days, e.g. 3 for predicting if Length of Stay is greater or equal to 3 days 
    Readmission_number_of_days_threshold: int = 30  # in days, e.g. 30 for predicting readmission within 30 days 
    Phenotype: str = "HF"  # HF, COPD, CKD, CAD 
    Phenotype_prediction_horizon: int = 24  # in hours, e.g. 24 for predicting phenotype within the first 24 hours of the ICU stay 

    
    Include_ICU: bool = True
    Include_Diagnosis: bool = True
    Include_Procedures: bool = True
    Include_Medications: bool = True
    Include_chart_event: bool = True 
    Include_output_event: bool = True 
    
    Disease_Filter: str = "None"  # None, HF, COPD, CKD, CAD 
    Outliers_management: str = "impute"  # remove, impute , No_outlier_detection 
    Outliers_threshold_right: float = 98.0  # right treshold for outlier detection, e.g. 98.0 for the 98th percentile  
    Outliers_threshold_left: float = 0.0  # left treshold for outlier detection, e.g. 0.0 for the 0th percentile (no left outliers)

    Time_window_size: int = 24  # in hours, e.g. 24 for first 24 hours 
    Time_window_bucket_size: int = 1  # in hours, e.g. 1 for hourly buckets 

    Missing_values_management: str = "FF_mean"  # FF_mean, FF_median, No_imputation 

    Oversampling: bool = True 
    Concatenate: bool = False
    
    Output_format: str = "csv"  # csv, pkl , npy

    def __post_init__(self):
        """Automatically validate config values after initialization."""
        self._validate()

    def _validate(self):
        errors = []

        valid_tasks = {"Mortality", "Length of stay", "Readmission", "Phenotype"}
        valid_phenotypes = {"HF", "COPD", "CKD", "CAD"}
        valid_disease_filters = {"None", "HF", "COPD", "CKD", "CAD"}
        valid_outlier_methods = {"remove", "impute","No_outlier_detection"}
        valid_missing = {"FF_mean", "FF_median", "No_imputation"}
        valid_formats = {"csv", "pkl", "npy"}

        if self.Task not in valid_tasks:
            errors.append(f"Invalid Task '{self.Task}'. Must be one of {valid_tasks}")

        if self.Phenotype not in valid_phenotypes:
            errors.append(f"Invalid Phenotype '{self.Phenotype}'. Must be one of {valid_phenotypes}")

        if self.Disease_Filter not in valid_disease_filters:
            errors.append(f"Invalid Disease_Filter '{self.Disease_Filter}'. Must be one of {valid_disease_filters}")

        if self.Outliers_management not in valid_outlier_methods:
            errors.append(
                f"Invalid Outliers_management '{self.Outliers_management}'. "
                f"Must be one of {valid_outlier_methods}"
            )

        if self.Missing_values_management not in valid_missing:
            errors.append(
                f"Invalid Missing_values_management '{self.Missing_values_management}'. "
                f"Must be one of {valid_missing}"
            )

        if self.Output_format not in valid_formats:
            errors.append(
                f"Invalid Output_format '{self.Output_format}'. "
                f"Must be one of {valid_formats}"
            )

        if self.Time_window_bucket_size > self.Time_window_size:
            errors.append(
                "Time_window_bucket_size cannot be larger than Time_window_size"
            )

        if not (0 <= self.Outliers_threshold_left <= 100):
            errors.append("Outliers_threshold_left must be between 0 and 100")

        if not (0 <= self.Outliers_threshold_right <= 100):
            errors.append("Outliers_threshold_right must be between 0 and 100")

        if self.Outliers_threshold_left > self.Outliers_threshold_right:
            errors.append(
                "Outliers_threshold_left cannot be greater than Outliers_threshold_right"
            )

        if not isinstance(self.Include_ICU, bool):
            errors.append("Include_ICU must be a boolean")

        if errors:
            print("\nConfig validation errors:")
            for e in errors:
                print(" -", e)

        if self.Concatenate == True: 
            # warning
            print("Warning: Concatenate is set to True, this may lead to very wide feature vectors and potential memory issues. Make sure this is intentional.")

    def save_to_json(self, path: str | None = None):
        """Saves this config at `path` if provided, else in the same place as `self.out_path`"""
        if path is None:
            path, _ = splitext(self.out_path)

        super().save_to_json(path + ".json")

    def load_from_json(self, path: str | None = None):
        """Loads this config from `path` if provided, else from the same place as `self.out_path`"""
        if path is None:
            path, _ = splitext(self.out_path)

        super().load_from_json(path)

    # def print 
    def __str__(self) -> str:
        """String representation of this config"""
        config_dict = asdict(self)
        config_str = "PreprocessConfig:\n"
        for key, value in config_dict.items():
            config_str += f"  {key}: {value}\n"
        return config_str

    def __repr__(self) -> str:
        """String representation of this config"""
        return self.__str__()



    
