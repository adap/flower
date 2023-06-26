# example_simulation_ray

This code splits CIFAR-10 dataset into `pool_size` partitions (user defined) and does a few rounds of CIFAR-10 training. In this example, we leverage [`Ray`](https://docs.ray.io/en/latest/index.html) to simulate Flower Clients participating in FL rounds in an resource-aware fashion. This is possible via the [`RayClientProxy`](https://github.com/adap/flower/blob/main/src/py/flwr/simulation/ray_transport/ray_client_proxy.py) which bridges a standard Flower server with standard Flower clients while excluding the gRPC communication protocol and the Client Manager in favour of Ray's scheduling and communication layers.

## Requirements

- Flower 0.18.0
- A recent version of PyTorch. This example has been tested with Pytorch 1.7.1, 1.8.2 (LTS) and 1.10.2.
- A recent version of Ray. This example has been tested with Ray 1.4.1, 1.6 and 1.9.2.

From a clean virtualenv or Conda environment with Python 3.7+, the following command will isntall all the dependencies needed:

```bash
$ pip install -r requirements.txt
```

# How to run

This example:

1. Downloads CIFAR-10
2. Partitions the dataset into N splits, where N is the total number of
   clients. We refere to this as `pool_size`. The partition can be IID or non-IID
3. Starts a Ray-based simulation where a % of clients are sample each round.
   This example uses N=10, so 10 clients will be sampled each round.
4. After the M rounds end, the global model is evaluated on the entire testset.
   Also, the global model is evaluated on the valset partition residing in each
   client. This is useful to get a sense on how well the global model can generalise
   to each client's data.

The command below will assign each client 2 CPU threads. If your system does not have 2xN(=10) = 20 threads to run all 10 clients in parallel, they will be queued but eventually run. The server will wait until all N clients have completed their local training stage before aggregating the results. After that, a new round will begin.

```bash
$ python main.py --num_client_cpus 2 # note that `num_client_cpus` should be <= the number of threads in your system.
```
