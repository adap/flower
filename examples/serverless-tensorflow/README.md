---
tags: [quickstart, vision, fds, serverless]
dataset: [CIFAR-10]
framework: [tensorflow]
---

# Serverless Federated Learning with Tensorflow/Keras and Flower (Quickstart Example)

This introductory example to Flower uses Tensorflow/Keras but deep knowledge of this frameworks is required to run the example. However, it will help you understand how to adapt Flower to your use case.
This example uses a partitioned CIFAR-10 dataset designed to test the performance of federated learning. There is artificial skew in the data distribution across nodes. For example, data on one node contains mostly classes 0-4, while data on the other node contains mostly classes 5-9.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/serverless-tensorflow . \
        && rm -rf _tmp \
        && cd serverless-tensorflow
```

This will create a new directory called `serverless-tensorflow`.

### Install dependencies and project

Install the `flwr` on the serverless branch:

```bash
pip install git+https://github.com/zzsi/flower.git@serverless
```

## Run the project

```
python run_simulation.py
```

The simulation requires 24GB of GPU memory. On an A10 GPU, it takes about 10 minutes to run.

