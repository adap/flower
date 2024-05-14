# Flower containers testing

This directory is used to test Flower containers in a minimum scenario, that is, with 2 clients and without HTTPS. The FL setup uses PyTorch, the CIFAR10 dataset, and a CNN. This is mainly to test the functionalities of Flower in a containerized architecture. It can be easily extended to test more complex communication set-ups.

It uses a subset of size 1000 for the training data and 10 data points for the testing.

To execute locally, run the following in CLI:
``` shell
$ docker compose up -d --remove-orphans --force-recreate --build
```

To stop the containers, run:
``` shell
$ docker compose down
```