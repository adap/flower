"""examplefkm: A Flower / Lifelines app."""

from lifelines.datasets import load_waltons

from flwr_datasets.partitioner import NaturalIdPartitioner
from datasets import Dataset

X = load_waltons()


def load_partition(partition_id: int):
    partitioner = NaturalIdPartitioner(partition_by="group")
    partitioner.dataset = Dataset.from_pandas(X)
    partition = partitioner.load_partition(partition_id).to_pandas()
    times = partition["T"].values
    events = partition["E"].values
    return times, events
