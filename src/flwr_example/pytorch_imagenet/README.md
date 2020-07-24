# PyTorch ImageNet Example

Install dependencies:

```shell
git clone git@github.com:adap/flower.git
cd flower
./dev/bootstrap.sh
```

## Start server

```shell
./src/flwr_example/pytorch_imagenet/run-server.sh
```

## Start two clients

To start the first client, open a new terminal and run:

```shell
./src/flwr_example/pytorch_imagenet/run-client.sh
```

To start the second client, open a new terminal and run:

```shell
./src/flwr_example/pytorch_imagenet/run-server.sh
```
