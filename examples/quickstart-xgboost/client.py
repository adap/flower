import warnings
import xgboost as xgb

import flwr as fl
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

from dataset import init_higgs, load_partition, split_train_test


warnings.filterwarnings("ignore", category=UserWarning)

# Load (HIGGS) dataset and conduct partitioning
num_partitions = 20
split_method = "uniform"
fds = init_higgs(num_partitions, "uniform")

# let's use the first partition as an example
partition_id = 0
partition = load_partition(fds, partition_id)

# train/test splitting and data re-formatting
SEED = 42
split_rate = 0.2
train_data, val_data = split_train_test(partition, split_rate, SEED)

# Hyper-parameters for training
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
        for i in range(num_local_round):
            self.bst.update(train_data, self.bst.num_boosted_rounds())

        # extract the last N=num_local_round trees as new local model
        bst = self.bst[self.bst.num_boosted_rounds() - num_local_round: self.bst.num_boosted_rounds()]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        if not self.bst:
            # first round local training
            print("Start training at round 1")
            bst = xgb.train(
                params,
                train_data,
                num_boost_round=num_local_round,
                evals=[(val_data, "validate"), (train_data, "train")],
            )
            self.config = bst.save_config()
            self.bst = bst
        else:
            print("load global model")
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
            num_examples=0,
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        eval_results = self.bst.eval_set(
            evals=[(train_data, "train"), (val_data, "valid")], iteration=self.bst.num_boosted_rounds() - 1
        )
        auc = round(float(eval_results.split("\t")[2].split(":")[1]), 4)

        global_round = ins.config["global_round"]
        print(f"AUC = {auc} at round {global_round}")

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=1,
            metrics={"AUC": auc},
        )


# Start Flower client
fl.client.start_client(server_address="127.0.0.1:8080", client=FlowerClient().to_client())

