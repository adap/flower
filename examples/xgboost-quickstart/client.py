import argparse
import warnings
from typing import Union
from logging import INFO
from datasets import Dataset, DatasetDict
import xgboost as xgb

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    Status,
)
from flwr_datasets.partitioner import IidPartitioner


warnings.filterwarnings("ignore", category=UserWarning)

# Define arguments parser for the client/partition ID.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--partition-id",
    default=0,
    type=int,
    help="Partition ID used for the current client.",
)
args = parser.parse_args()


# Define data partitioning related functions
def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data


# Load (HIGGS) dataset and conduct partitioning
# We use a small subset (num_partitions=30) of the dataset for demonstration to speed up the data loading process.
partitioner = IidPartitioner(num_partitions=30)
fds = FederatedDataset(dataset="jxie/higgs", partitioners={"train": partitioner})

# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
partition = fds.load_partition(partition_id=args.partition_id, split="train")
partition.set_format("numpy")

# Train/test splitting
train_data, valid_data, num_train, num_val = train_test_split(
    partition, test_fraction=0.2, seed=42
)

# Reformat data to DMatrix for xgboost
log(INFO, "Reformatting data...")
train_dmatrix = transform_dataset_to_dmatrix(train_data)
valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

# Hyper-parameters for xgboost training
num_local_round = 1
params = {
    "objective": "binary:logistic",
    "eta": 0.1,  # Learning rate
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


# Define Flower client
class XgbClient(fl.client.Client):
    def __init__(
        self,
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
            )
        else:
            bst = xgb.Booster(params=self.params)
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        # Load global model
        bst = xgb.Booster(params=self.params)
        for para in ins.parameters.tensors:
            para_b = bytearray(para)
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc},
        )


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=XgbClient(
        train_dmatrix,
        valid_dmatrix,
        num_train,
        num_val,
        num_local_round,
        params,
    ).to_client(),
)
