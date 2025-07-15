# Flower with Tensorflow and multiple strategies testing

This directory is used for testing Flower with Tensorflow by using the CIFAR10 dataset and a CNN.

It uses a subset of size 1000 for the training data and 10 data points for the testing.

It tests the following strategies:

- FedMedian
- FedTrimmedAvg
- QFedAvg
- FaultTolerantFedAvg
- FedAvgM
- FedAdam
- FedAdagrad
- FedYogi
