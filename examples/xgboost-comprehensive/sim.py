import warnings
from logging import INFO
import xgboost as xgb
from tqdm import tqdm

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from dataset import (
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
    separate_xy,
    resplit,
)
from utils import (
    sim_args_parser,
    NUM_LOCAL_ROUND,
    BST_PARAMS,
)
from server_utils import (
    eval_config,
    fit_config,
    evaluate_metrics_aggregation,
    get_evaluate_fn,
    CyclicClientManager,
)
from client_utils import XgbClient


warnings.filterwarnings("ignore", category=UserWarning)


def get_client_fn(
    train_data_list, valid_data_list, train_method, params, num_local_round
):
    """Return a function to construct a client.

    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """

    def client_fn(cid: str) -> fl.client.Client:
        """Construct a FlowerClient with its own dataset partition."""
        x_train, y_train = train_data_list[int(cid)][0]
        x_valid, y_valid = valid_data_list[int(cid)][0]

        # Reformat data to DMatrix
        train_dmatrix = xgb.DMatrix(x_train, label=y_train)
        valid_dmatrix = xgb.DMatrix(x_valid, label=y_valid)

        # Fetch the number of examples
        num_train = train_data_list[int(cid)][1]
        num_val = valid_data_list[int(cid)][1]

        # Create and return client
        return XgbClient(
            train_dmatrix,
            valid_dmatrix,
            num_train,
            num_val,
            num_local_round,
            params,
            train_method,
        )

    return client_fn


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
    if args.centralised_eval or args.centralised_eval_client:
        log(INFO, "Loading centralised test set...")
        test_data = fds.load_split("test")
        test_data.set_format("numpy")
        num_test = test_data.shape[0]
        test_dmatrix = transform_dataset_to_dmatrix(test_data)

    # Load partitions and reformat data to DMatrix for xgboost
    log(INFO, "Loading client local partitions...")
    train_data_list = []
    valid_data_list = []

    # Load and process all client partitions. This upfront cost is amortized soon
    # after the simulation begins since clients wont need to preprocess their partition.
    for partition_id in tqdm(range(args.pool_size), desc="Extracting client partition"):
        # Extract partition for client with partition_id
        partition = fds.load_partition(partition_id=partition_id, split="train")
        partition.set_format("numpy")

        if args.centralised_eval_client:
            # Use centralised test set for evaluation
            train_data = partition
            num_train = train_data.shape[0]
            x_test, y_test = separate_xy(test_data)
            valid_data_list.append(((x_test, y_test), num_test))
        else:
            # Train/test splitting
            train_data, valid_data, num_train, num_val = train_test_split(
                partition, test_fraction=args.test_fraction, seed=args.seed
            )
            x_valid, y_valid = separate_xy(valid_data)
            valid_data_list.append(((x_valid, y_valid), num_val))

        x_train, y_train = separate_xy(train_data)
        train_data_list.append(((x_train, y_train), num_train))

    # Define strategy
    if args.train_method == "bagging":
        # Bagging training
        strategy = FedXgbBagging(
            evaluate_function=(
                get_evaluate_fn(test_dmatrix) if args.centralised_eval else None
            ),
            fraction_fit=(float(args.num_clients_per_round) / args.pool_size),
            min_fit_clients=args.num_clients_per_round,
            min_available_clients=args.pool_size,
            min_evaluate_clients=(
                args.num_evaluate_clients if not args.centralised_eval else 0
            ),
            fraction_evaluate=1.0 if not args.centralised_eval else 0.0,
            on_evaluate_config_fn=eval_config,
            on_fit_config_fn=fit_config,
            evaluate_metrics_aggregation_fn=(
                evaluate_metrics_aggregation if not args.centralised_eval else None
            ),
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
    # In this example we use CPU by default
    client_resources = {
        "num_cpus": args.num_cpus_per_client,
        "num_gpus": 0.0,
    }

    # Hyper-parameters for xgboost training
    num_local_round = NUM_LOCAL_ROUND
    params = BST_PARAMS

    # Setup learning rate
    if args.train_method == "bagging" and args.scaled_lr:
        new_lr = params["eta"] / args.pool_size
        params.update({"eta": new_lr})

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(
            train_data_list,
            valid_data_list,
            args.train_method,
            params,
            num_local_round,
        ),
        num_clients=args.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_manager=CyclicClientManager() if args.train_method == "cyclic" else None,
    )


if __name__ == "__main__":
    main()
