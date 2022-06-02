# User Guide

Our implementation extends the Flower framework with a client and a strategy
that support differentially private (DP) federated learning for PyTorch models
by integrating with the Opacus package.

## Installing dependencies

1. Install Python
2. `pip install -r requirements.txt`

## Running the code

On Linux and Mac, you can run a server and two clients as subprocesses using:

`python main.py`

Or you can use the bash script: `./run.sh`

On Windows, you will need to open three terminal windows.

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
