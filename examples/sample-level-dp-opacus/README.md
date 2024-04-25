# Training with Sample-Level Differential Privacy using Opacus Privacy Engine

In this example, we demonstrate how to train a model with differential privacy (DP) using Flower. We employ PyTorch and integrate the Opacus Privacy Engine to achieve sample-level differential privacy. This setup ensures robust privacy guarantees during the client training phase. The code is adapted from the [PyTorch Quickstart example](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch).

For more information about DP in Flower please refer to the [tutorial](https://flower.ai/docs/framework/how-to-use-differential-privacy.html). For additional information about Opacus, visit the official \[website\] (https://opacus.ai/).

## Environments Setup

Start by cloning the example. We prepared a single-line command that you can copy into your shell which will checkout the example for you:

```shell
git clone --depth=1 https://github.com/adap/flower.git && mv flower/examples/sample-level-dp-opacus . && rm -rf flower && cd sample-level-dp-opacus
```

This will create a new directory called `sample-level-dp-opacus` containing the following files:

```shell
-- requirements.txt
-- client.py
-- server.py
-- README.md
```

### Installing dependencies

Project dependencies are defined in `requirements.txt`. Install them with:

```shell
pip install -r requirements.txt
```

## Run Flower with Opacus and Pytorch

You can simply start the server in a terminal as follows:

```shell
python3 server.py
```

Now, you're ready to start the Flower clients that will participate in the learning process. We need to specify the partition id to utilize different partitions of the data across various nodes. Additionally, you can specify the DP hyperparameters (in this example, they are `target-delta`, `noise-multiplier`, and `max-grad-norm` with default values of 1e-5, 1.3, and 1.0) for each client. To do so, simply open two more terminal windows and execute the following commands.

Start client 1 in the first terminal:

```shell
python3 client.py --partition-id 0 --noise-multiplier 1.5
```

Start client 2 in the second terminal:

```shell
python3 client.py --partition-id 1 --noise-multiplier 1.1
```

You can observe the computed privacy budget for each client on every round.
