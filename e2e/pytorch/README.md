# Flower with PyTorch testing

This directory is used for testing Flower with PyTorch by using the CIFAR10 dataset and a CNN.

It uses the `FedAvg` strategy with an `evaluate_metrics_aggregation_fn` provided.

It uses a subset of size 1000 for the training data and 10 data points for the testing.