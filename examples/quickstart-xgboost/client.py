import warnings
import argparse
from logging import INFO
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

from dataset import (
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
)


warnings.filterwarnings("ignore", category=UserWarning)


def args_parser():
    """Parse arguments to define experimental settings."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num_partitions", default=20, type=int, help="Number of partitions."
    )
    parser.add_argument(
        "--partitioner_type",
        default="uniform",
        type=str,
        choices=["uniform", "linear", "square", "exponential"],
        help="Partitioner types.",
    )
    parser.add_argument(
        "--partition_id",
        default=0,
        type=int,
        help="Partition ID used for the current client.",
    )
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed used for train/test splitting."
    )
    parser.add_argument(
        "--test_fraction",
        default=0.2,
        type=float,
        help="test fraction for train/test splitting.",
    )

    args_ = parser.parse_args()
    return args_


# Parse arguments for experimental settings
args = args_parser()

# Load (HIGGS) dataset and conduct partitioning
num_partitions = args.num_partitions
# partitioner type is chosen from ["uniform", "linear", "square", "exponential"]
partitioner_type = args.partitioner_type

# instantiate partitioner
partitioner = instantiate_partitioner(
    partitioner_type=partitioner_type, num_partitions=num_partitions
)
fds = FederatedDataset(dataset="jxie/higgs", partitioners={"train": partitioner})

# let's use the first partition as an example
partition_id = args.partition_id
partition = fds.load_partition(idx=partition_id, split="train")
partition.set_format("numpy")

# train/test splitting and data re-formatting
SEED = args.seed
test_fraction = args.test_fraction
train_data, valid_data, num_train, num_val = train_test_split(
    partition, test_fraction=test_fraction, seed=SEED
)

# reformat data to DMatrix for xgboost
train_dmatrix = transform_dataset_to_dmatrix(train_data)
valid_dmatrix = transform_dataset_to_dmatrix(valid_data)


# Hyper-parameters for xgboost training
num_local_round = 1
params = {
    "objective": "binary:logistic",
    "eta": 0.1,  # lr
    "max_depth": 8,
    "eval_metric": "auc",
    "nthread": 16,
    "num_parallel_tree": 1,
    "subsample": 1,
    "tree_method": "hist",
}


# Define Flower client
class FlowerClient(fl.client.Client):
    def __init__(self):
        self.bst = None

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def local_boost(self):
        # update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        # extract the last N=num_local_round trees as new local model
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_round : self.bst.num_boosted_rounds()
        ]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # first round local training
            log(INFO, "Start training at round 1")
            bst = xgb.train(
                params,
                train_dmatrix,
                num_boost_round=num_local_round,
                evals=[(valid_dmatrix, "validate"), (train_dmatrix, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            log(INFO, "Load global model")
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self.local_boost()

        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=num_train,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(train_dmatrix, "train"), (valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[2].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        log(INFO, f"AUC = {auc} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=num_val,
            metrics={"AUC": auc},
        )


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080", client=FlowerClient().to_client()
)
