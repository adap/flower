"""$project_name: A Flower / XGBoost app."""

import xgboost as xgb
from flwr.client import Client, ClientApp
from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    GetParametersRes,
    Parameters,
    Status,
)

from $project_name.task import load_data


# Define Flower Client and client_fn
class FlowerClient(Client):
    def __init__(self, train_dmatrix, valid_dmatrix, num_train, num_val):
        self.bst = None
        self.config = None
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_local_round = 1
        self.params = {
            "objective": "binary:logistic",
            "eta": 0.1,  # Learning rate
            "max_depth": 8,
            "eval_metric": "auc",
            "nthread": 16,
            "num_parallel_tree": 1,
            "subsample": 1,
            "tree_method": "hist",
        }
        self.num_train = num_train
        self.num_val = num_val

    def get_parameters(self, ins):
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
        )

    def _local_boost(self):
        # Update trees based on local training data.
        for i in range(self.num_local_round):
            self.bst.update(self.train_dmatrix, self.bst.num_boosted_rounds())

        # Extract the last N=num_local_round trees for sever aggregation
        bst = self.bst[
            self.bst.num_boosted_rounds()
            - self.num_local_round : self.bst.num_boosted_rounds()
        ]

        return bst

    def fit(self, ins):
        if not self.bst:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.valid_dmatrix, "validate"), (self.train_dmatrix, "train")],
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
            num_examples=self.num_train,
            metrics={},
        )

    def evaluate(self, ins):
        eval_results = self.bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        auc = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_val,
            metrics={"AUC": auc},
        )

def client_fn(cid):
    # Load model and data
    train_dmatrix, valid_dmatrix, num_train, num_val = load_data(int(cid), 2)

    # Return Client instance
    return FlowerClient(train_dmatrix, valid_dmatrix, num_train, num_val).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
