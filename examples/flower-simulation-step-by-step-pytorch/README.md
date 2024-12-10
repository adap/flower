---
tags: [basic, vision, simulation]
dataset: [MNIST]
framework: [torch]
---

# Flower Simulation Step-by-Step

This directory contains the code developed in the `Flower Simulation 2025` tutorial series on Youtube. You can find all the videos [here](https://www.youtube.com/playlist?list=PLNG4feLHqCWkdlSrEL2xbCtGa6QBxlUZb) or clicking on the video preview below.

<div align="center">
      <a href="https://www.youtube.com/playlist?list=PLNG4feLHqCWkdlSrEL2xbCtGa6QBxlUZb" target="_blank" rel="noopener noreferrer">
         <img src="https://img.youtube.com/vi/XK_dRVcSZqg/0.jpg" style="width:75%;">
      </a>
</div>

## Complementary Resources

In this tutorial series, we make reference to several pages in the [Flower Documentation](https://flower.ai/docs/). In particular, this videos highlight pages for:

- [Visualizing Dataset Distributions using `flwr-datasets`](https://flower.ai/docs/datasets/tutorial-visualize-label-distribution.html)
- [List of all Partitioners available in `flwr-datasets`](https://flower.ai/docs/datasets/ref-api/flwr_datasets.partitioner.html)
- [How-to Run Simulations page](https://flower.ai/docs/framework/how-to-run-simulations.html)
- [How-to Design Stateful ClientApps](https://flower.ai/docs/framework/how-to-design-stateful-clients.html)
- [Advanced PyTorch Example](https://github.com/adap/flower/tree/main/examples/advanced-pytorch)

## Getting Started

> \[!TIP\]
> If you are developing on Windows, it is recommended to make use of the Windows Subsystem for Linux (WSL). Check the guide on [how to setup WSL for development on Windows](https://code.visualstudio.com/docs/remote/wsl).

> \[!NOTE\]
> These steps represent the very first commands show on the first video. They `flwr new` command will create a Flower App you can run directly. In videos 2-7 you'll learn how to modify the App and add, step by step, the functionality already present in the `my-awesome-app` diredtory in this directory. You can use it as reference.

As presented in the video, we start from a new Python 3.11 environment. You only need to activate it and install `flwr`.

```bash
# Install Flower
pip install -U flwr
```

Then, use the `flwr new` command to construct a new Flower App using the PyTorch template:

```shell
flwr new my-awesome-app # then follow the prompt
```

## Running the Example

Just like all other Flower Apps, you can run the one in this directory by means of `flwr run`. More info about this command in the videos!

```shell
flwr run my-awesome-app
```

The output you should expect without making changes to the code is as follows:

```shell
Loading project configuration...
Success
wandb: Using wandb-core as the SDK backend.  Please refer to https://wandb.me/wandb-core for more information.
wandb: Currently logged in as: <your-user>. Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.18.7
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run custom-strategy-2024-12-10_07:32:03
wandb: ⭐️ View project at https://wandb.ai/<your-user>/flower-simulation-tutorial
wandb: 🚀 View run at https://wandb.ai/<your-user>/flower-simulation-tutorial/runs/reyoryuu
INFO :      Starting Flower ServerApp, config: num_rounds=3, no round_timeout
INFO :
INFO :      [INIT]
INFO :      Using initial global parameters provided by strategy
INFO :      Starting evaluation of initial global parameters
INFO :      initial parameters (loss, other metrics): 2.3028839167695456, {'cen_accuracy': 0.0937}
INFO :
INFO :      [ROUND 1]
INFO :      configure_fit: strategy sampled 5 clients (out of 10)
INFO :      aggregate_fit: received 5 results and 0 failures
INFO :      fit progress: (1, 2.0274660648248446, {'cen_accuracy': 0.3238}, 5.769022958003916)
INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
INFO :      aggregate_evaluate: received 10 results and 0 failures
INFO :
INFO :      [ROUND 2]
INFO :      configure_fit: strategy sampled 5 clients (out of 10)
INFO :      aggregate_fit: received 5 results and 0 failures
INFO :      fit progress: (2, 0.7511614774362728, {'cen_accuracy': 0.6926}, 11.233382292004535)
INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
INFO :      aggregate_evaluate: received 10 results and 0 failures
INFO :
INFO :      [ROUND 3]
INFO :      configure_fit: strategy sampled 5 clients (out of 10)
INFO :      aggregate_fit: received 5 results and 0 failures
INFO :      fit progress: (3, 0.5243101176172019, {'cen_accuracy': 0.8035}, 13.289899208000861)
INFO :      configure_evaluate: strategy sampled 10 clients (out of 10)
INFO :      aggregate_evaluate: received 10 results and 0 failures
INFO :
INFO :      [SUMMARY]
INFO :      Run finished 3 round(s) in 13.60s
INFO :          History (loss, distributed):
INFO :                  round 1: 2.0251417029128924
INFO :                  round 2: 0.7533456925429649
INFO :                  round 3: 0.5141592433326874
INFO :          History (loss, centralized):
INFO :                  round 0: 2.3028839167695456
INFO :                  round 1: 2.0274660648248446
INFO :                  round 2: 0.7511614774362728
INFO :                  round 3: 0.5243101176172019
INFO :          History (metrics, distributed, fit):
INFO :          {'max_b': [(1, 0.8776450829832974),
INFO :                     (2, 0.8755706409526767),
INFO :                     (3, 0.880116537616749)]}
INFO :          History (metrics, distributed, evaluate):
INFO :          {'accuracy': [(1, 0.3237817576009996),
INFO :                        (2, 0.6909620991253644),
INFO :                        (3, 0.8046647230320699)]}
INFO :          History (metrics, centralized):
INFO :          {'cen_accuracy': [(0, 0.0937), (1, 0.3238), (2, 0.6926), (3, 0.8035)]}
INFO :
wandb: 🚀 View run custom-strategy-2024-12-10_07:32:03 at: https://wandb.ai/<your-user>/flower-simulation...
wandb: Find logs at: wandb/run-20241210_073204-reyoryuu/logs
```
