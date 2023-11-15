---
title: FedBN: Federated Learning on Non-IID Features via Local Batch Normalization
url: https://arxiv.org/abs/2102.07623
labels: [label1, label2] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [MNIST, MNIST-M, SVHN, USPS, SynthDigits]
---

# :warning: FedBN: Federated Learning on Non-IID Features via Local Batch Normalization

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.


**Paper:** [arxiv.org/abs/2102.07623](https://arxiv.org/abs/2102.07623)

**Authors:** Xiaoxiao Li, Meirui Jiang, Xiaofei Zhang, Michael Kamp, Qi Dou

**Abstract:** The emerging paradigm of federated learning (FL) strives to enable collaborative training of deep models on the network edge without centrally aggregating raw data and hence improving data privacy. In most cases, the assumption of independent and identically distributed samples across local clients does not hold for federated learning setups. Under this setting, neural network training performance may vary significantly according to the data distribution and even hurt training convergence. Most of the previous work has focused on a difference in the distribution of labels or client shifts. Unlike those settings, we address an important problem of FL, e.g., different scanners/sensors in medical imaging, different scenery distribution in autonomous driving (highway vs. city), where local clients store examples with different distributions compared to other clients, which we denote as feature shift non-iid. In this work, we propose an effective method that uses local batch normalization to alleviate the feature shift before averaging models. The resulting scheme, called FedBN, outperforms both classical FedAvg, as well as the state-of-the-art for non-iid data (FedProx) on our extensive experiments. These empirical results are supported by a convergence analysis that shows in a simplified setting that FedBN has a faster convergence rate than FedAvg.


## About this baseline

**What’s implemented:** Figure 3 in the paper: convergence in training loss comparing `FedBN` to `FedAvg` for five datasets.

**Datasets:** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset)._*

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

**Task:** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

**Model:** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

**Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

To construct the Python environment, simply run:

```bash
# Set directory to use python 3.10 (install with `pyenv install <version>` if you don't have it)
pyenv local 3.10.6

# Tell poetry to use python3.10
poetry env use 3.10.6

# Install
poetry install
```

Before running the experiments you'll need to download the five datasets for this baseline. We'll be using the pre-processed datasets created by the `FedBN` authors. Download the dataset from [here](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155149226_link_cuhk_edu_hk/EV1YgHfFC4RKjw06NL4JMdgBMU21CegM12SpXrSmGjt3XA?e=XK2rFs) and move the file into a new directory named `data`.
```bash
mkdir data
mv <path/to/your/digit_dataset.zip> data/

# now uncompress the zipfile
cd data && unzip digit_dataset.zip
cd data ..
```

## Running the Experiments

First, activate your environment via `poetry shell`. The commands below show how to run the experiments and modify some of its key hyperparameters via the cli. Each time you run an experiment, the log and results will be stored inside `outputs/<date>/<time>`.

```bash
# run with default arguments
python -m fedbn.main

# if you only care about clients doing fit() and not evaluate(),
# you can disable that stage like so:
python -m fedbn.main strategy.fraction_evaluate=0 # your code will run many times faster

# adjust hyperparameters like the number of rounds or batch size like this
python -m fedbn.main num_rounds=100 dataset.batch_size

# increase the number of clients like this (note this should be a multiple 
# of the number of dataset you involve in the experiment -- 5 by default)
# this means that without changing other hyperparemeters, you can only have
# either 5,10,15,20,25,30,35,40,45 or 50 clients
python -m fedbn.main num_clients=20

# by default clients get assigned a 10th of the data in a dataset
# this is because the datasets you downloaded were pre-processed by the authors
# of the FedBN paper. They created 10 partitions.
# You can increase the amount of partitions each client gets by increasing dataset.percent
# PLease note that as you increase that value, the maximum number of clients
# you can have in your experiment gets reduced (this is because partitions are fixed and
# can't be -- unless you add support for it -- partitioned into smaller ones)
python -m fedbn.main dataset.percent=0.2 # max allowed is 25 clients

# run with FedAvg clients leaving the rest default
python -m fedbn.main client=fedavg
```

## Limitations

dataset-number of clients relationship

## Expected Results

Replicate the results shown below by running the following command. First ensureing you have activated your environment.

```bash

python -m fedbn.main --multirun num_rounds=100 client=fedavg,fedbn

# then use the notebook in docs/multirun_plot.ipynb to create the plot below
```

![FedBn vs FedAvg on all datasets](_static/train_loss.png)