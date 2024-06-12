# Federated Instruction Tuning (static)
---
dataset:
  name: $dataset_name

# FL experimental settings
num_clients: $num_clients # total number of clients
num_rounds: 200
partitioner:
  _target_: flwr_datasets.partitioner.IidPartitioner
  num_partitions: $num_clients
