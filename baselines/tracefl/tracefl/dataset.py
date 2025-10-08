"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from tracefl.dataset_preparation import ClientsAndServerDatasets


def medical_dataset2labels(dname):
    """Retrieve the mapping of class indices to label names for a given dataset."""
    if dname == "pathmnist":
        return {
            0: "Adipose",
            1: "Background",
            2: "Debris",
            3: "Lymphocytes",
            4: "Mucus",
            5: "Smooth Muscle",
            6: "Normal Colon Mucosa",
            7: "Cancer-associated Stroma",
            8: "Colorectal Adenocarcinoma",
        }
    return None


def fix_dataset_keys(dataset):
    """Fix dataset keys to ensure consistent format.

    This function ensures that all dataset keys follow a consistent format by
    converting them to strings and adding a 'client_' prefix if needed.

    Args:
        dataset: Dictionary containing dataset information

    Returns
    -------
        Dictionary with fixed keys
    """
    if "client2data" in dataset and "clients_traindata" not in dataset:
        dataset["clients_traindata"] = dataset["client2data"]
    if "clients_testdata" not in dataset:
        dataset["clients_testdata"] = {}
    if "server_data" in dataset and "server_testdata" not in dataset:
        dataset["server_testdata"] = dataset["server_data"]
    return dataset


def get_clients_server_data(cfg):
    """Obtain datasets for clients and server based on the provided configuration.

    Returns
    -------
        dict: A dictionary containing the datasets. Expected keys are:
            - "clients_traindata": dict mapping client IDs to training data.
            - "clients_testdata": dict mapping client IDs to client test data.
            - "server_testdata": server's test data.
            - "client2class": mapping of client IDs to class counts.
            - "fds": the federated dataset object.
    """
    ds_prep = ClientsAndServerDatasets(cfg)
    ds_dict = ds_prep.get_data()
    ds_dict = fix_dataset_keys(ds_dict)

    for key in ["clients_traindata", "clients_testdata", "server_testdata"]:
        if key not in ds_dict:
            raise KeyError(
                f"Missing '{key}' in dataset. Available keys: {list(ds_dict.keys())}"
            )

    return ds_dict


def load_central_server_test_data(cfg):
    """Load the test data intended for the central server.

    Returns
    -------
        object: The test dataset for the central server.
    """
    d_obj = ClientsAndServerDatasets(cfg).get_data()
    d_obj = fix_dataset_keys(d_obj)
    return d_obj["server_testdata"]
