---
title: Federated Self-training for Semi-supervised Audio Recognition
url: https://dl.acm.org/doi/10.1145/3520128
labels: [Audio Classification, Semi Supervised learning]
dataset: [Ambient Context, Speech Commands]
---

# FedStar: Federated Self-training for Semi-supervised Audio Recognition

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:**  [dl.acm.org/doi/10.1145/3520128](https://dl.acm.org/doi/10.1145/3520128)

**Authors:** Vasileios Tsouvalas, Aaqib Saeed, Tanir Özcelebi

**Abstract:** Federated Learning is a distributed machine learning paradigm dealing with decentralized and personal datasets. Since data reside on devices such as smartphones and virtual assistants, labeling is entrusted to the clients or labels are extracted in an automated way. Specifically, in the case of audio data, acquiring semantic annotations can be prohibitively expensive and time-consuming. As a result, an abundance of audio data remains unlabeled and unexploited on users’ devices. Most existing federated learning approaches focus on supervised learning without harnessing the unlabeled data. In this work, we study the problem of semi-supervised learning of audio models via self-training in conjunction with federated learning. We propose FedSTAR to exploit large-scale on-device unlabeled data to improve the generalization of audio recognition models. We further demonstrate that self-supervised pre-trained models can accelerate the training of on-device models, significantly improving convergence within fewer training rounds. We conduct experiments on diverse public audio classification datasets and investigate the performance of our models under varying percentages of labeled and unlabeled data. Notably, we show that with as little as 3% labeled data available, FedSTAR on average can improve the recognition rate by 13.28% compared to the fully supervised federated model.


## About this baseline

**What’s implemented:** The code is structured in such a way that all experiments for ambient context and speech commands can be derived.

**Datasets:** Ambient Context, Speech Commands

**Hardware Setup:** These experiments were run on a linux server with 56 CPU threads with 325 GB Ram with A10 GPU in it. Any machine with 16 CPU cores and 32 GB memory would be able to run experiments with small number of clients in a reasonable amount of time. For context, a machine with 24 cores and a RTX3090Ti ran the Speech Commands experiment in Table 3 with 10 clients in 1h. For this experiment 30GB of RAM was used and clients required ~1.4GB of VRAM each. The same experiment but with the Ambient Context dataset too 13minutes.

**Contributors:** Raj Parekh [GitHub](https://github.com/Raj-Parekh24), [Mail](rajparekhwc@gmail.com)

## Environment Setup
```bash
# Set python version
pyenv local 3.10.6
# Tell poetry to use python 3.10
poetry env use 3.10.6
# Now install the environment
poetry install
# Start the shell to activate your environment.
poetry shell
```

Next, you'll need to download the datasets. In the case of SpeechCommands, some preprocessing is also required:

```bash
# Make the shell script executable
chmod +x setup_datasets.sh

# The below script will download the datasets and create a directory structure require to run this experiment.
./setup_datasets.sh

# If you want to run the SpeechCommands experiment, pre-process the dataset
# This will generate a few training example from the _silence_ category
python -m fedstar.dataset_preparation
# Please note the above will make following changes:
#    * Add new files to datasets/speech_commands/Data/Train/_silence_
#    * Add new entries to data_splits/speech_commands/train_split.txt
# Therefore the above command should only be run once. If you want to run it again
# after making modifications to the script, please either revert the changes outlined
# above or erase the dataset and repeat the download + preprocessing as defined in setup_datasets.sh script.
```

## Setting up GPU Memory

**Note:** The experiment is designed to run on both GPU and CPU, but runs better on a system with GPU (specially when using the SpeechCommands dataset). If you wish to use GPU, make sure you have installed the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads). This baseline has been tested with CUDA 12.3. By default, it will run only on the CPU. Please update the value of the list `gpu_total_mem` with the corresponding memory for each GPU in your machine that you want to expose to the experiment. The variable is in the `distribute_gpus` function inside the `clients.py`. Reference is shown below.

```python
# For Eg:- We have a system with two GPUs with 8GB and 4GB VRAM.
#          The modified variable will looks like below.
gpu_free_mem = [8000,4000]
```



## Running the Experiments

By default, the `Ambient Context` experiment in Table 3 with 10 clients will be run.

```bash
python -m fedstar.server
python -m fedstar.clients
```

You can change the dataset, number of clients and number of rounds like this:

```bash
python -m fedstar.server num_clients=5 dataset_name=speech_commands server.rounds=20
python -m fedstar.clients num_clients=5 dataset_name=speech_commands
```

To run experiments for Table 4, you should pass a different config file (i.e. that in `fedstar/conf/table4.yaml`). You can do this as follows:

```bash
# by default will run FedStar with Ambient Context and L=3%
python -m fedstar.server --config-name table4
python -m fedstar.clients --config-name table4
```

To modify the ratio of labelled data do so as follows:
```bash
# To use a different L setting
python -m fedstar.server --config-name table4 L=L5 # {L3, L5, L20, L50}
# same for fedstar.clients
```

To run in supervised mode, pass `fedstar=false` to any of the commands above (when launching both the server and clients). Naturally, you can also override any other setting, like `dataset_name` and `num_clients` if desired.


## Expected Results


This section indicates the commands to execute to obtain the results shown below in Table 3 and Table 4. While both configs fix the number of rounds to 100, in many settings fewer rounds are enough for the model to reach the accuracy shown in the tables. The commands below make use of Hydra's `--multirun` to run multiple experiments. This is better suited when using Flower simulations. Here they work fine but, if you encounter any issues, you can always "unroll" the multirun and run one configuration at a time. If you do this, results won't go into the `multirun/` directory, instead to the default `outputs/` directory.


### Table 3

Results will be stored in `multirun/Table3/<dataset_name>/N_<num_clients>/<date>/<time>`. Please note since we are running two Hydra processes, both server and client will generate a log and therefore respective subdirectories in `multirun/`. This is a small compromise of not using Flower simulation. 

```bash
# For Ambient Context
python -m fedstar.server --multirun num_clients=5,10,15,30
python -m fedstar.clients --multirun num_clients=5,10,15,30

# For SpeechCommands
python -m fedstar.server --multirun num_clients=5,10,15,30 dataset_name=speech_commands
python -m fedstar.clients --multirun num_clients=5,10,15,30 dataset_name=speech_commands
```

| Clients | Speech Commands |                | Ambient Context |                |
|---------|-----------------|----------------|-----------------|----------------|
|         | Actual          | Implementation | Actual          | Implementation |
| N=5     | 96.93           | 97.15          | 71.88           | 72.60          |
| N=10    | 96.78           | 96.42          | 68.01           | 68.43          |
| N=15    | 96.33           | 96.43          | 66.86           | 66.28          |
| N=30    | 94.62           | 95.37          | 65.14           | 59.45          |


### Table 4

Following the logic presented for obtaining Table 3 results, the larger Table 4 set of results can be obtained by running the `--multirun` commands shown below.

```bash
# Generate supervised results for Ambient Context (note this will run 4x4=16 experiments)
python -m fedstar.server --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50 fedstar=false
python -m fedstar.clients --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50 fedstar=false

# Generate supervised results for Speech Commands (note this will run 4x4=16 experiments)
python -m fedstar.server --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50 dataset_name=speech_commands fedstar=false
python -m fedstar.clients --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50 dataset_name=speech_commands fedstar=false

# Generate FedStar results for Ambient Context
python -m fedstar.server --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50
python -m fedstar.clients --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50

# Generate FedStar results for Speech Commands
python -m fedstar.server --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50 dataset_name=speech_commands
python -m fedstar.clients --config-name table4 --multirun num_clients=5,10,15,30 L=L3,L5,L20,L50 dataset_name=speech_commands
```



|Dataset|Clients|SupervisedFederatedLearning<br>L=3&#8195;&#8195;L=5&#8195;&#8195;L=20&#8195;&#8195;L=50&#8195;&#8195;L=100|FedStar<br>L=3&#8195;&#8195;L=5&#8195;&#8195;L=20&#8195;&#8195;L=50|
|:--:|:--:|:---:|:---:|
Ambient<br></p>Context|5<br>10<br>15<br>30|43.75&#8195;&#8195;45.17&#8195;&#8195;63.40&#8195;&#8195;67.57&#8195;75.27<br>41.75&#8195;&#8195;44.90&#8195;&#8195;56.23&#8195;&#8195;60.82&#8195;70.39<br>43.18&#8195;&#8195;42.75&#8195;&#8195;49.60&#8195;&#8195;59.47&#8195;68.41<br>36.44&#8195;&#8195;38.73&#8195;&#8195;48.91&#8195;&#8195;55.83&#8195;67.23|49.60&#8195;&#8195;52.78&#8195;&#8195;66.12&#8195;&#8195;66.71<br>47.84&#8195;&#8195;52.48&#8195;&#8195;61.98&#8195;&#8195;63.46<br>49.05&#8195;&#8195;56.05&#8195;&#8195;64.25&#8195;&#8195;64.05<br>47.34&#8195;&#8195;46.51&#8195;&#8195;60.45&#8195;&#8195;55.47|
Speech<br></p>Commands|5<br>10<br>15<br>30|80.83&#8195;&#8195;87.97&#8195;&#8195;92.35&#8195;&#8195;94.66&#8195;95.87<br>54.66&#8195;&#8195;78.90&#8195;&#8195;92.13&#8195;&#8195;93.71&#8195;96.55<br>53.44&#8195;&#8195;64.53&#8195;&#8195;91.83&#8195;&#8195;94.23&#8195;96.35<br>36.41&#8195;&#8195;42.87&#8195;&#8195;83.68&#8195;&#8195;93.18&#8195;94.24|87.39&#8195;&#8195;90.11&#8195;&#8195;94.09&#8195;&#8195;94.85<br>87.15&#8195;&#8195;91.06&#8195;&#8195;94.74&#8195;&#8195;96.15<br>86.78&#8195;&#8195;91.01&#8195;&#8195;95.21&#8195;&#8195;95.70<br>80.89&#8195;&#8195;85.62&#8195;&#8195;94.08&#8195;&#8195;94.18|
