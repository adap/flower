This directory contains the code developed in the `Flower Simulation` tutorial series on Youtube. You can find all the videos [here](https://www.youtube.com/playlist?list=PLNG4feLHqCWlnj8a_E1A_n5zr2-8pafTB) or clicking on the video preview below.

- In `Part-I` (7 videos) we developed from scratch a complete Federated Learning pipeline for simulation using PyTorch.
- In `Part-II` (2 videos) we _enhanced_ the code in `Part-I` by making a better use of Hydra configs.

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
