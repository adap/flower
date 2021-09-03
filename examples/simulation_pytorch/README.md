# example_simulation_ray

This code splits CIFAR-10 dataset into `pool_size` partitions (user defined) and does a few rounds of CIFAR-10 training.

## Requirements

*    Flower nightly release (or development version from `main` branch)
*    PyTorch 1.7.1 (but most likely will work with older versions)
*    Ray 1.4.1

# How to run

This example:

1. Downloads CIFAR-10
2. Partitions the dataset into N splits, where N is the total number of
   clients. We refere to this as `pool_size`. The partition can be IID or non-IID
4. Starts a Ray-based simulation where a % of clients are sample each round.
5. After the M rounds end, the global model is evaluated on the entire testset.
   Also, the global model is evaluated on the valset partition residing in each
   client. This is useful to get a sense on how well the global model can generalise
   to each client's data.

```bash
$ python main.py
```
