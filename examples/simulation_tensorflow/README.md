# example_simulation_ray

This code splits CIFAR-10 dataset into `pool_size` partitions (user defined) and performs a few rounds of CIFAR-10 training.

## Requirements

*    Flower nightly release (or development version from `main` branch)
*    Tensorflow 2.6.8 (or newer)
*    Ray 1.11.0

# How to run

This example will:

1. Download CIFAR-10
2. Partition the dataset into N splits, where N is the total number of
   clients. We refer to this as `pool_size`. The partition can be IID or non-IID
4. Starts a Ray-based simulation where a % of clients are sample each round.
5. After the M rounds end, the global model is evaluated on the entire testset.
   Also, the global model is evaluated on the valset partition residing in each
   client. This is useful to get a sense on how well the global model can generalise
   to each client's data.

```bash
$ python main.py
```
