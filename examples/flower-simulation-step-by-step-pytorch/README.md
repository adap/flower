---
title: Flower Simulation Step-by-Step
url: https://pytorch.org/
labels: [basic, vision, simulation]
dataset: [MNIST]
framework: [torch]
---

# Flower Simulation Step-by-Step

> Since this tutorial (and its video series) was put together, Flower has been updated a few times. As a result, some of the steps to construct the environment (see below) have been updated. Some parts of the code have also been updated. Overall, the content of this tutorial and how things work remains the same as in the video tutorials.

This directory contains the code developed in the `Flower Simulation` tutorial series on Youtube. You can find all the videos [here](https://www.youtube.com/playlist?list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB) or clicking on the video preview below.

- In `Part-I` (7 videos) we developed from scratch a complete Federated Learning pipeline for simulation using PyTorch.
- In `Part-II` (2 videos) we _enhanced_ the code in `Part-I` by making a better use of Hydra configs.

<div align="center">
      <a href="https://www.youtube.com/playlist?list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB" target="_blank" rel="noopener noreferrer">
         <img src="https://img.youtube.com/vi/cRebUIGB5RU/0.jpg" style="width:75%;">
      </a>
</div>

## Constructing your Python Environment

As presented in the video, we first need to create a Python environment. You are free to choose the tool you are most familiar with, we'll be using `conda` in this tutorial. You can create the conda and setup the environment as follows:

```bash
# I'm assuming you are running this on an Ubuntu 22.04 machine (GPU is not required)

# create the environment
conda create -n flower_tutorial python=3.9 -y

# activate your environment (depending on how you installed conda you might need to use `conda activate ...` instead)
source activate flower_tutorial

# install PyToch (other versions would likely work)
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia -y
# conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch -y # If you don't have a GPU

# Install Flower and other dependencies
pip install -r requirements.txt
```

If you are running this on macOS with Apple Silicon (i.e. M1, M2), you'll need a different `grpcio` package if you see an error when running the code. To fix this do:

```bash
# with your conda environment activated
pip uninstall grpcio

conda install grpcio -y
```
