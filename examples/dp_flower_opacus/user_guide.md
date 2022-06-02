# User Guide

Our implementation extends the Flower framework with a client and a strategy
that support differentially private federated learning for PyTorch models by integrating with the Opacus package.

## Installing dependencies

1. Install Python
2. `pip install -r requirements.txt`

## Running the code

On Linux and Mac, you can run a server and two clients as subprocesses using:

`python main.py`

On Windows, you will need to open three terminal windows.

In the first, type: `python server.py`

In the second, type: `python client.py 0`

In the third, type: `python client.py 1`

The training progress, loss, accuracy, and epsilon metrics will be logged to the
terminal.
