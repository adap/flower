import warnings
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
    resplit,
)
from utils import client_args_parser


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
args = client_args_parser()

# Load (HIGGS) dataset and conduct partitioning
num_partitions = args.num_partitions

# Partitioner type is chosen from ["uniform", "linear", "square", "exponential"]
partitioner_type = args.partitioner_type

# Instantiate partitioner
partitioner = instantiate_partitioner(
    partitioner_type=partitioner_type, num_partitions=num_partitions
)
fds = FederatedDataset(
    dataset="jxie/higgs", partitioners={"train": partitioner}, resplitter=resplit
)

# Let's use the first partition as an example
node_id = args.node_id
partition = fds.load_partition(idx=node_id, split="train")
partition.set_format("numpy")

if args.centralised_eval:
    # Use centralised test set for evaluation
    train_data = partition
    valid_data = fds.load_full("test")
    valid_data.set_format("numpy")
    num_train = train_data.shape[0]
    num_val = valid_data.shape[0]
else:
    # Train/test splitting
    SEED = args.seed
    test_fraction = args.test_fraction
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=test_fraction, seed=SEED
    )

# Reformat data to DMatrix for xgboost
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
class FlowerClient(fl.client.Client):
    def __init__(self):
        self.bst = None
        self.config = None

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        _ = (self, ins)
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(num_local_round):
            self.bst.update(train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # First round local training
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
            for item in ins.parameters.tensors:
                global_model = bytearray(item)

            # Load global model into booster
            self.bst.load_model(global_model)
            self.bst.load_config(self.config)

            bst = self._local_boost()

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
            evals=[(valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
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
            num_examples=num_val,
            metrics={"AUC": auc},
        )


# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient())
