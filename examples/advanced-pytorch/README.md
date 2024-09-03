---
tags: [advanced, vision, fds, wandb]
dataset: [Fashion-MNIST]
framework: [torch, torchvision]
---

> \[!TIP\]
> This example shows intermediate and advanced functionality of Flower. It you are new to Flower, it is recommended to start from the [quickstart-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example or the [quickstart PyTorch tutorial](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)

# Federated Learning with PyTorch and Flower (Advanced Example)

This example extends the content of the [quickstart-pytorch](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) example. Primarily it demonstrates how to customize your `ServerApp` so it gains additional functionality:

- Saves results (e.g. accuracy, loss) to a JSON in the file system
- Saves a checkpoint of the global model when a new best is found
- Logs metrics to [Weight&Biases](<>) if enabled
- _Stateful clients_: `ClientApp` make use of ther context to persist metrics at the clients across participation rounds.

This examples shows how to achieve the above using both a `Strategy` as well as the low-level API.

## Set up the project

### Clone the project

Start by cloning the example project:

```shell
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/advanced-pytorch . \
        && rm -rf _tmp \
        && cd advanced-pytorch
```

This will create a new directory called `advanced-pytorch` with two sub-directories, each cotaining the same Flower app but one makes use of high-level components such as the `strategy` and the other uses the low-level API.

### Run the examples

Please navigate to either sub-directory to run the example.
