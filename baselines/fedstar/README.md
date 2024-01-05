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

**Hardware Setup:** These experiments were run on a linux server with 56 CPU threads with 325 GB Ram with A10 GPU in it. Any machine with 8 CPU cores and 16 GB memory or more would be able to run it in a reasonable amount of time. Note: I have install tensorflow with GPU support but by default, the entire experiment runs on CPU-only mode. For cpu you need to replace value of gpus to None present in distribute_gpus function inside clients.py file.

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

# The below script will download the datasets and create a directory structure requir to run this experiment.
./setup_datasets.sh

# If you want to run the SpeechCommands experiment, pre-process the dataset
# This will genereate a few training example from the _silence_ category
python -m fedstar.silence_processing
# Please note the above will make following changes:
#    * Add new files to datasets/speech_commands/Data/Train/_silence_
#    * Add new entries to data_splits/speech_commands/train_split.txt
# Therefore the above command should only be run once. If you want to run it again
# after making modifications to the script, please either revert the changes outlined
# above or erase the dataset and repeat the download + preprocessing as defined in setup_datasets.sh script.
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

To run experiments for Table 4, you should pass a different cofig file (i.e. that in `fedstar/conf/table4.yaml`). You can do this as follows:

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
This section indicates the commands to exectue to obtain the results shown below in Table 3 and Table 4. The commands below make use of Hydra's `--multirun` to run multiple experiments. This is better suited when using Flower simulations. Here they work fine but, if you encounter any issues, you can always "unroll" the multirun and run one configuration at a time. If you do this, results won't go into the `multirun/` directory, instead to the default `outputs/` directory.

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
| N=5     | 96.93           | 95.83          | 71.88           | 73.13          |
| N=10    | 96.78           | 96.21          | 68.01           | 70.56          |
| N=15    | 96.33           | 96.72          | 66.86           | 66.28          |
| N=30    | 94.62           | 94.51          | 65.14           | 66.17          |


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



| Dataset | Clients | Supervised Federated Learning <br> L=3% &#8195; &#8195; L=5%  &#8195; L=20%  &#8195; L=50% &#8195; L=100% | FedStar <br> L=3% &#8195; &#8195; L=5%  &#8195; L=20%  &#8195; L=50% | 
| :--: | :--:| :---: | :---: |
Ambient Context | 5<br> 10<br> 15 <br>30 | 48.23 &#8195; &#8195; 49.65 &#8195; &#8195; 65.22 &#8195; &#8195; 68.23 &#8195; 75.27 <br> 34.61 &#8195; &#8195; 44.29 &#8195; &#8195; 54.67 &#8195; &#8195; 62.51 &#8195; 70.39 <br> 33.46 &#8195; &#8195; 44.11 &#8195; &#8195; 52.31 &#8195; &#8195; 62.58 &#8195; 68.41 <br> 33.12 &#8195; &#8195; 39.88 &#8195; &#8195; 50.93 &#8195; &#8195; 59.46 &#8195; 67.23 | 50.21 &#8195; &#8195; 55.13 &#8195; &#8195; 67.58 &#8195; &#8195; 69.53 <br> 49.37 &#8195; &#8195; 49.51 &#8195; &#8195; 64.31 &#8195; &#8195; 68.31 <br> 49.28 &#8195; &#8195; 57.23 &#8195; &#8195; 64.87 &#8195; &#8195; 65.46 <br> 42.16 &#8195; &#8195; 47.32 &#8195; &#8195; 62.96 &#8195; &#8195; 55.98 |
Speech Commands | 5<br> 10<br> 15 <br>30 | 80.83 &#8195; &#8195; 87.97 &#8195; &#8195; 92.35 &#8195; &#8195; 94.66 &#8195; 96.93 <br> 67.75 &#8195; &#8195; 82.97 &#8195; &#8195; 92.13 &#8195; &#8195; 94.02 &#8195; 96.55 <br> 62.90 &#8195; &#8195; 72.16 &#8195; &#8195; 92.14 &#8195; &#8195; 93.06 &#8195; 96.35 <br> 33.53 &#8195; &#8195; 44.21 &#8195; &#8195; 84.93 &#8195; &#8195; 92.13 &#8195; 94.58 | 87.39 &#8195; &#8195; 90.11 &#8195; &#8195; 94.09 &#8195; &#8195; 94.85 <br> 86.73 &#8195; &#8195; 90.33 &#8195; &#8195; 94.12 &#8195; &#8195; 94.14 <br> 86.78 &#8195; &#8195; 89.34 &#8195; &#8195; 93.15 &#8195; &#8195; 93.23 <br> 83.86 &#8195; &#8195; 88.17 &#8195; &#8195; 92.91 &#8195; &#8195; 92.59 |
