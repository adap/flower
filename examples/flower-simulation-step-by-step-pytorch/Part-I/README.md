# A Complete FL Simulation Pipeline using Flower

In the first part of the Flower Simulation series, we go step-by-step through the process of designing a FL pipeline. Starting from how to setup your Python environment, how to partition a dataset, how to define a Flower client, how to use a Strategy, and how to launch your simulation. The code in this directory is the one developed in the video. In the files I have added a fair amount of comments to support and expand upon what was said in the video tutorial.

# Link to step-by-step videos

This step-by-step tutorial is organized as a 1+9 videos Youtube series. The first video introduces the tutorial and provides an outline for the next 9 videos. You can start the tutorial by clicking on the video thumbnail below.

<div align="center">
      <a href="https://www.youtube.com/playlist?list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB">
         <img src="https://img.youtube.com/vi/cRebUIGB5RU/0.jpg" style="width:75%;">
      </a>
</div>

# Constructing your Python Environment

As presented in the video, we first need to create a Python environment. You are free to choose the tool you are most familiar with, we'll be using `conda` in this tutorial. You can create the conda and setup the environment as follows:

```bash
# I'm assuming you are running this on an Ubuntu 22.04 machine (GPU is not required)

# create the environment
conda create -n flower_tutorial python=3.8 -y

# activate your environment (depending on how you installed conda you might need to use `conda activate ...` instead)
source activate flower_tutorial

# install PyToch (other versions would likely work)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch # If you don't have a GPU

# install flower (for FL) and hydra (for configs)
pip install flwr==1.4.0 hydra-core==1.3.2
# install ray
# you might see some warning messages after installing it (you can ignore them)
pip install ray==1.11.1
```

If you are running this on macOS with Apple Silicon (i.e. M1, M2), you'll need a different `grpcio` package if you see an error when running the code. To fix this do:
```bash
# with your conda environment activated
pip uninstall grpcio

conda install grpcio -y
```

# Running the Code

In this tutorial we didn't dive in that much into Hydra configs (that's the content of [Part-II](https://github.com/adap/flower/tree/main/examples/flower-simulation-step-by-step-pytorch/Part-II)). However, this doesn't mean we can't easily configure our experiment directly from the command line. Let's see a couple of examples on how to run our simulation.

```bash

# this will launch the simulation using default settings
python main.py

# you can override the config easily for instance
python main.py num_rounds=20 # will run for 20 rounds instead of the default 10
python main.py config_fit.lr=0.1 # will use a larger learning rate for the clients.
```
