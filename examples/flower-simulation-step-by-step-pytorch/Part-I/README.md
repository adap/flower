# A Complete FL Simulation Pipeline using Flower

In the first part of the Flower Simulation series, we go step-by-step through the process of designing a FL pipeline. Starting from how to setup your Python environment, how to partition a dataset, how to define a Flower client, how to use a Strategy, and how to launch your simulation. The code in this directory is the one developed in the video. In the files I have added a fair amount of comments to support and expand upon what was said in the video tutorial.

## Running the Code

In this tutorial we didn't dive in that much into Hydra configs (that's the content of [Part-II](https://github.com/adap/flower/tree/main/examples/flower-simulation-step-by-step-pytorch/Part-II)). However, this doesn't mean we can't easily configure our experiment directly from the command line. Let's see a couple of examples on how to run our simulation.

```bash

# this will launch the simulation using default settings
python main.py

# you can override the config easily for instance
python main.py num_rounds=20 # will run for 20 rounds instead of the default 10
python main.py config_fit.lr=0.1 # will use a larger learning rate for the clients.
```
