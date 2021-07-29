# Differentially Private Federated Learning using Opacus, PyTorch and Flower

This example contains code demonstrating how to include the Opacus library for training a model using DP-SGD. The code is adapted from multiple other examples:

- PyTorch Quickstart
- Simulation Quickstart
- Simulation Extended Example

## Requirements

- **Flower** nightly release (or development version from `main` branch) for the simulation, otherwise normal Flower for the client 
- **PyTorch** 1.7.1 (but most likely will work with older versions)
- **Ray** 1.4.1 (just for the simulation)
- **Opacus** 0.14.0

## Privacy Parameters

The parameters can be set in `dp_cifar_main.py`.

## Running the client

Run the server with `python server.py`. Then open two (or more) new terminals to start two (or more) clients with `python dp_cifar_client.py`.

## Running the simulation

Note: It is not possible to see the total privacy budget used with this example since the simulation creates clients from scratch every round.

Run the simulation with `python dp_cifar_simulation.py`.

