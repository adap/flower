"""pandas_example: A Flower / Pandas app."""

import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

from flwr.client import ClientApp
from flwr.common import Context


from flwr.client import ClientApp
from flwr.common import Message, RecordSet, MetricsRecord, Context


fds = None  # Cache FederatedDataset


def get_clientapp_dataset(partition_id: int, num_partitions: int):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="scikit-learn/iris",
            partitioners={"train": partitioner},
        )

    dataset = fds.load_partition(partition_id, "train").with_format("pandas")[:]
    # Use just the specified columns
    return dataset[["SepalLengthCm", "SepalWidthCm"]]


# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    dataset = get_clientapp_dataset(partition_id, num_partitions)

    metrics = {}
    # Compute histogram for each column in dataframe
    for feature_name in dataset.columns:
        freqs, _ = np.histogram(dataset[feature_name], bins=10)
        metrics[feature_name] = freqs.tolist()

    reply_content = RecordSet(metrics_records={"query_results": MetricsRecord(metrics)})

    return msg.create_reply(reply_content)
