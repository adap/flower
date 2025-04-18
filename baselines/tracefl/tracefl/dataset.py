"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

import logging

from tracefl.dataset_preparation import ClientsAndServerDatasets


def medical_dataset2labels(dname):
    """Retrieve the mapping of class indices to label names for a given
    dataset.
    """
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
    else:
        return None


def fix_dataset_keys(ds_dict):
    """Convert keys from the current dataset dictionary into the keys expected
    by TraceFL.

    Expected keys by TraceFL:
      - "clients_traindata": training data for each client.
      - "clients_testdata": test data for each client.
      - "server_testdata": test data for the central server.
    """
    if "client2data" in ds_dict and "clients_traindata" not in ds_dict:
        ds_dict["clients_traindata"] = ds_dict["client2data"]
    if "clients_testdata" not in ds_dict:
        ds_dict["clients_testdata"] = {}
    if "server_data" in ds_dict and "server_testdata" not in ds_dict:
        ds_dict["server_testdata"] = ds_dict["server_data"]
    return ds_dict


def get_clients_server_data(cfg):
    """Obtain datasets for clients and server based on the provided
    configuration.

    Returns
    -------
        dict: A dictionary containing the datasets. Expected keys are:
            - "clients_traindata": dict mapping client IDs to training data.
            - "clients_testdata": dict mapping client IDs to client test data.
            - "server_testdata": server's test data.
            - "client2class": mapping of client IDs to class counts.
            - "fds": the federated dataset object.
    """
    logging.info("Creating dataset from scratch (no cache)...")
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


def convert_client2_faulty_client(ds, label2flip, target_label_col="label"):
    """Transform a client's dataset to simulate a faulty client by flipping
    specified labels.

    Parameters
    ----------
        ds (Dataset): The dataset to modify.
        label2flip (dict): Mapping of original label values to flipped values.
        target_label_col (str, optional): Key for the label in each example.

    Returns
    -------
        dict: A dictionary with keys:
            - 'ds': The transformed dataset.
            - 'label2count': A mapping of each label to its count in the transformed dataset.
    """

    def flip_label(example):
        label = example[target_label_col]
        if hasattr(label, "item"):
            label = label.item()
        if label in label2flip:
            example[target_label_col] = label2flip[label]
        return example

    ds = ds.map(flip_label).with_format("torch")
    label2count = {}
    for example in ds:
        label = example[target_label_col].item()
        label2count[label] = label2count.get(label, 0) + 1

    return {"ds": ds, "label2count": label2count}
