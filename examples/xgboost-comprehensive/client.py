import warnings
from logging import INFO

import flwr as fl
from flwr_datasets import FederatedDataset
from flwr.common.logger import log

from dataset import (
    instantiate_partitioner,
    train_test_split,
    transform_dataset_to_dmatrix,
    resplit,
)
from utils import client_args_parser, BST_PARAMS, NUM_LOCAL_ROUND
from client_utils import XgbClient


warnings.filterwarnings("ignore", category=UserWarning)


# Parse arguments for experimental settings
args = client_args_parser()

# Train method (bagging or cyclic)
train_method = args.train_method

# Load (HIGGS) dataset and conduct partitioning
# Instantiate partitioner from ["uniform", "linear", "square", "exponential"]
partitioner = instantiate_partitioner(
    partitioner_type=args.partitioner_type, num_partitions=args.num_partitions
)
fds = FederatedDataset(
    dataset="jxie/higgs",
    partitioners={"train": partitioner},
    preprocessor=resplit,
)

# Load the partition for this `partition_id`
log(INFO, "Loading partition...")
partition = fds.load_partition(partition_id=args.partition_id, split="train")
partition.set_format("numpy")

if args.centralised_eval:
    # Use centralised test set for evaluation
    train_data = partition
    valid_data = fds.load_split("test")
    valid_data.set_format("numpy")
    num_train = train_data.shape[0]
    num_val = valid_data.shape[0]
else:
    # Train/test splitting
    train_data, valid_data, num_train, num_val = train_test_split(
        partition, test_fraction=args.test_fraction, seed=args.seed
    )

# Reformat data to DMatrix for xgboost
log(INFO, "Reformatting data...")
train_dmatrix = transform_dataset_to_dmatrix(train_data)
valid_dmatrix = transform_dataset_to_dmatrix(valid_data)

# Hyper-parameters for xgboost training
num_local_round = NUM_LOCAL_ROUND
params = BST_PARAMS

# Setup learning rate
if args.train_method == "bagging" and args.scaled_lr:
    new_lr = params["eta"] / args.num_partitions
    params.update({"eta": new_lr})

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
        train_method,
    ),
)
