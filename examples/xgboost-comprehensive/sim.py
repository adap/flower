import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log
from flwr.server.strategy import FedXgbBagging, FedXgbCyclic

from dataset import (
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
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


def get_client_fn(fds, args, params, num_local_round):
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

    # Hyper-parameters for xgboost training
    num_local_round = NUM_LOCAL_ROUND
    params = BST_PARAMS

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=get_client_fn(fds, args, params, num_local_round),
        num_clients=args.pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_manager=CyclicClientManager() if args.train_method == "cyclic" else None,
    )


if __name__ == "__main__":
    main()
