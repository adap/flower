---
title: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization (NeurIPS 2020) \
url: https://proceedings.neurips.cc/paper/2020/hash/564127c03caab942e503ee6f810f54fd-Abstract.html \
labels: [normalized averaging, heterogeneous optimization, federated learning]  \
dataset: [non-iid cifar10 dataset, synthetic dataset]
---

# FedNova: Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization

****Paper:**** [arxiv.org/abs/2007.07481](https://arxiv.org/abs/2007.07481)

****Authors:**** *Jianyu Wang, Qinghua Liu, Hao Liang, Gauri Joshi, H. Vincent Poor*

****Abstract:**** *In federated learning, heterogeneity in the clients' local datasets and computation speeds results in large variations in the number of local updates performed by each client in each communication round. Naive weighted aggregation of such models causes objective inconsistency, that is, the global model converges to a stationary point of a mismatched objective function which can be arbitrarily different from the true objective. This paper provides a general framework to analyze the convergence of federated heterogeneous optimization algorithms. It subsumes previously proposed methods such as FedAvg and FedProx and provides the first principled understanding of the solution bias and the convergence slowdown due to objective inconsistency. Using insights from this analysis, we propose FedNova, a normalized averaging method that eliminates objective inconsistency while preserving fast error convergence.*


## About this baseline

****What’s implemented:**** *The code in this baseline aims to reproduce the results from Fig 5 in the paper which corresponds to Synthetic(1,1) experiment and further Table 1 for CIFAR 10 dataset.*

****Datasets:**** *_This basleline experiments tackles heterogeneous data sources. Non-IID CIFAR-10 dataset and Synthetic(1,1) dataset(creation methodology detailed in [link](https://arxiv.org/pdf/1812.06127.pdf) are used in the experiments._*

****Hardware Setup:**** *_The baseline comprises of federated learning on 16 clients. If we want to run each client in parallel, it would require 16 cpu cores and roughly ~16 GB of GPU memory for a batch size of 32. 
However, most experiments were performed using a pool of 6 actors(6 clients run in parallel out of 16) requiring 6 cpu cores and 6GB of GPU memory. 
The experiments were performed on a A100 machine however any GPU with 6GB of memory and 6 cpu cores should be sufficiently fast._*

****Contributors:**** *_Aasheesh Singh (Github: @ashdtu), MILA-Quebec AI Institute_*


## Experimental Setup

****Task:**** :warning: *CIFAR-10 dataset --> Image classification_*

****Model:**** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

****Dataset:**** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

****Training Hyperparameters:**** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

``` python
# Set python version
pyenv install 3.10.6
pyenv local 3.10.6

# Tell poetry to use python 3.10
poetry env use 3.10.6

# install the base Poetry environment
poetry install

# activate the environment
poetry shell
```

## Running the Experiments

:warning: _Provide instructions on the steps to follow to run all the experiments._
```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

poetry run -m <baseline-name>.main <no additional arguments> # where <baseline-name> is the name of this directory and that of the only sub-directory in this directory (i.e. where all your source code is)

# If you are using a dataset that requires a complicated download (i.e. not using one natively supported by TF/PyTorch) + preprocessing logic, you might want to tell people to run one script first that will do all that. Please ensure the download + preprocessing can be configured to suit (at least!) a different download directory (and use as default the current directory). The expected command to run to do this is:

poetry run -m <baseline-name>.dataset_preparation <optional arguments, but default should always run>

# It is expected that you baseline supports more than one dataset and different FL settings (e.g. different number of clients, dataset partitioning methods, etc). Please provide a list of commands showing how these experiments are run. Include also a short explanation of what each one does. Here it is expected you'll be using the Hydra syntax to override the default config.

poetry run -m <baseline-name>.main  <override_some_hyperparameters>
.
.
.
poetry run -m <baseline-name>.main  <override_some_hyperparameters>
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

``` bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
