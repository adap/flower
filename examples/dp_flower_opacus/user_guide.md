# User Guide

Our implementation extends the Flower framework with a client and a strategy
that support differentially private (DP) federated learning for PyTorch models
by integrating with the Opacus package.

## Installing dependencies

1. Install Python (3.8 or higher) from [python.org](https://www.python.org/downloads/)
2. Open a terminal window and type `pip install -r requirements.txt` to install
   dependency packages.

## Running the code

On Linux and Mac, you can run a server and two clients as subprocesses using:

`python main.py` or `python3 main.py` depending on your system configuration.

This will run federated learning with one server and two clients with default
arguments. To see all available command line arguments: `python main.py --help`.

Or you can run the demo using the provided bash script: `./run.sh`

On Windows, you will need to open three terminal windows to start a server and
two clients:

- In the first, type: `python server.py`
- In the second, type: `python client.py 0`
- In the third, type: `python client.py 1`

The training progress, loss, accuracy, and epsilon metrics will be logged to the
terminal.

## Code organization

- client.py is a command-line interface for starting a Flower DP client for
  demonstration purposes.
- dp_client.py contains the DPClient class, which integrates the Opacus DP
  framework.
- fedavgdp.py contains FedAvgDp, a subclass of the FedAvg Strategy that excludes
  and disconnects clients that have exceeded their privacy budget.
- main.py contains example code for creating a PyTorch neural network model and
  performing federated training with a benchmark dataset using DPClient and
  FedAvgDP.
- server.py is a command-line interface for starting a Flower DP server for
  demonstration purposes.

## Using this in your own project

To use this functionality in your own federated learning project:

1. Import the `FedAvgDp` class, construct an instance, and pass it as the
  `strategy` argument to the `flwr.server.start_server` function.
2. Import the `DPClient` class, construct an instance, providing the required
   arguments (a PyTorch module, optimizer, and train and test data loaders, and
   an Opacus `PrivacyEngine` instance, and the desired `target_epsilon` and
   `target_delta` values), and then pass this instance as the `client` argument
   to `flwr.client.start_numpy_client`.
