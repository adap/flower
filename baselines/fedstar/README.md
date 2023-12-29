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
# Please not the above will:
#    * Add new files to datasets/speech_commands/Data/Train/_silence_
#    * Add new entries to data_splits/speech_commands/train_split.txt
# Therefore the above command should only be run once. If you want to run it again
# after making modifications to the script, please either revert the changes outlinedo
# above or erase the dataset and repeat the download + preprocessing
```

## Running the Experiments

By default, the `Ambient Context` experiment in Table 3 with 10 clients will be run.

```bash
python -m fedstar.server
python -m fedstar.clients
```

You can change the dataset and number of clients like this:

```bash
python -m fedstar.server num_clients=5 dataset_name=speech_commands
# the use the same settings when launching fedstar.clients
```


## Expected Results
### Table 3
| Clients | Speech Commands |                | Ambient Context |                |
|---------|-----------------|----------------|-----------------|----------------|
|         | Actual          | Implementation | Actual          | Implementation |
| N=5     | 96.93           | 95.83          | 71.88           | 73.13          |
| N=10    | 96.78           | 96.21          | 68.01           | 70.56          |
| N=15    | 96.33           | 96.72          | 66.86           | 66.28          |
| N=30    | 94.62           | 94.51          | 65.14           | 66.17          |


### Table 4
| Dataset | Clients | Supervised Federated Learnning | | |  |  |  |  |  | |  |  | Fedstar | | | | | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  | L=3% |  | L=5% |  | L=20% |  | L=50% |  | L=100% |  |  | L=3% |  | L=5% |  | L=20% |  | L=50% |  |
|  |  | Actual | Implementation | Actual | Implementation | Actual | Implementation | Actual | Implementation | Actual | Implementation |  | Actual | Implementation | Actual | Implementation | Actual | Implementation | Actual | Implementation |
| Ambient Context | 5.0 | 46.34 | 48.23 | 47.89 | 49.65 | 61.4 | 65.22 | 65.85 | 68.23 | 71.88 | 75.27 |  | 48.68 | 50.21 | 54.95 | 55.13 | 64.37 | 67.58 | 67.04 | 69.53 |
|  | 10.0 | 35.29 | 34.61 | 41.31 | 44.29 | 51.71 | 54.67 | 62.69 | 62.51 | 68.01 | 70.39 |  | 48.87 | 49.37 | 52.37 | 49.51 | 62.94 | 64.31 | 64.42 | 68.31 |
|  | 15.0 | 33.03 | 33.46 | 42.75 | 44.11 | 53.37 | 52.31 | 59.97 | 62.58 | 66.86 | 68.41 |  | 49.54 | 49.28 | 54.71 | 57.23 | 63.46 | 64.87 | 62.41 | 65.46 |
|  | 30.0 | 32.31 | 33.12 | 40.17 | 39.88 | 47.05 | 50.93 | 55.85 | 59.46 | 65.14 | 67.23 |  | 40.84 | 42.16 | 46.58 | 47.32 | 60.21 | 62.96 | 56.19 | 55.98 |
|  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Speech Commands | 5.0 | 81.12 | 76.56 | 87.97 | 84.53 | 92.35 | 87.58 | 94.66 | 88.24 | 96.93 | 89.52 |  | 87.41 | 81.55 | 90.01 | 84.32 | 94.17 | 88.63 | 94.85 | 89.36 |
|  | 10.0 | 67.75 | 62.84 | 83.8 | 79.32 | 92.12 | 88.37 | 94.02 | 89.38 | 96.78 | 89.31 |  | 86.82 | 80.93 | 90.33 | 86.17 | 94.09 | 87.92 | 94.18 | 89.57 |
|  | 15.0 | 62.98 | 60.33 | 72.84 | 68.47 | 92.14 | 86.95 | 93.14 | 87.02 | 96.33 | 88.87 |  | 86.82 | 81.52 | 89.33 | 85.59 | 93.16 | 86.35 | 93.39 | 87.52 |
|  | 30.0 | 33.78 | 28.47 | 44.21 | 39.96 | 84.94 | 79.88 | 92.21 | 86.49 | 94.62 | 87.55 |  | 83.88 | 78.55 | 88.19 | 82.26 | 92.92 | 85.87 | 92.62 | 86.63 |