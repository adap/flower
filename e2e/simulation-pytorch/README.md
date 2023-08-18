# Flower's VirtualClientEngine with PyTorch testing

This directory is used for testing Flower's VirtualClientEngine to simulate FL workloads with PyTorch by using the CIFAR10 dataset and a CNN. This test heavily borrows from that in [e2d/pytorch](https://github.com/adap/flower/tree/main/e2e/pytorch).

It uses the `FedAvg` strategy with a pool of 100 clients. The VCE allocates 1 cpu core per actor.

It uses a subset of size 1000 for the training data and 10 data points for the testing.