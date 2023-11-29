import warnings
from logging import INFO
import xgboost as xgb

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic
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
from utils import (
    sim_args_parser,
    BST_PARAMS,
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    CyclicClientManager,
)


warnings.filterwarnings("ignore", category=UserWarning)


# Hyper-parameters for xgboost training
num_local_round = 1
params = BST_PARAMS


def get_client_fn(fds, args):
    """Return a function to construct a client.

    The VirtualClientEngine will exectue this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""

        # Extract partition for client with node_id = cid
        partition = fds.load_partition(node_id=int(cid), split="train")
        partition.set_format("numpy")

        if args.centralised_eval_client:
            # Use centralised test set for evaluation
            train_data = partition
            valid_data = fds.load_full("test")
            valid_data.set_format("numpy")
            num_train = train_data.shape[0]
            num_val = valid_data.shape[0]
        else:
            # Train/test splitting
            train_data, valid_data, num_train, num_val = train_test_split(
                partition, test_fraction=args.test_fraction, seed=args.seed
            )

        # Reformat data to DMatrix for xgboost
        train_dmatrix = transform_dataset_to_dmatrix(train_data)
        valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

        # Create and return client
        return XgbClient(
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            args.train_method,
        )

    return client_fn


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
        train_method,
    ):
        self.train_dmatrix = train_dmatrix
        self.valid_dmatrix = valid_dmatrix
        self.num_train = num_train
        self.num_val = num_val
        self.num_local_round = num_local_round
        self.params = params
        self.train_method = train_method

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
        # Cyclic: return the entire model
        bst = (
            bst_input[
                bst_input.num_boosted_rounds()
                - self.num_local_round : bst_input.num_boosted_rounds()
            ]
            if self.train_method == "bagging"
            else bst_input
        )

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


def main():
    # Parse arguments for experimental settings
    args = sim_args_parser()

    # Load (HIGGS) dataset and conduct partitioning
    partitioner = instantiate_partitioner(
        partitioner_type=args.partitioner_type, num_partitions=args.pool_size
    )
    fds = FederatedDataset(
        dataset="jxie/higgs",
        partitioners={"train": partitioner},
        resplitter=resplit,
    )

    # Load centralised test set
    if args.centralised_eval:
        fds = FederatedDataset(
            dataset="jxie/higgs", partitioners={"train": 20}, resplitter=resplit
        )
        log(INFO, "Loading centralised test set...")
        test_set = fds.load_full("test")
        test_set.set_format("numpy")
        test_dmatrix = transform_dataset_to_dmatrix(test_set)

    # Define strategy
    if args.train_method == "bagging":
        # Bagging training
        strategy = FedXgbBagging(
            evaluate_function=get_evaluate_fn(test_dmatrix)
            if args.centralised_eval
            else None,
            fraction_fit=(float(args.num_clients_per_round) / args.pool_size),
            min_fit_clients=args.num_clients_per_round,
            min_available_clients=args.pool_size,
            min_evaluate_clients=args.num_evaluate_clients
            if not args.centralised_eval
            else 0,
            fraction_evaluate=1.0 if not args.centralised_eval else 0.0,
            on_evaluate_config_fn=eval_config,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation
            if not args.centralised_eval
            else None,
        )
    else:
        # Cyclic training
        strategy = FedXgbCyclic(
            fraction_fit=1.0,
            min_available_clients=args.pool_size,
            fraction_evaluate=1.0,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=eval_config,
            on_fit_config_fn=fit_config,
        )

    # Resources to be assigned to each virtual client
    # In this example we use CPU in default
    client_resources = {
        "num_cpus": args.num_cpus_per_client,
        "num_gpus": 0.0,
    }

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, args),
        num_clients=args.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_manager=CyclicClientManager() if args.train_method == "cyclic" else None,
    )


if __name__ == "__main__":
    main()
