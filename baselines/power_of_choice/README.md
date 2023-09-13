---
title: Towards Understanding Biased Client Selection in Federated Learning
url: https://proceedings.mlr.press/v151/jee-cho22a.html
labels: [client selection, dynamic selection, heterogeneous clients] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [FMNIST, CIFAR10] # list of datasets you include in your baseline
---

# Power of Choice

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

> :warning: This is the template to follow when creating a new Flower Baseline. Please follow the instructions in `EXTENDED_README.md`

> :warning: Please follow the instructions carefully. You can see the [FedProx-MNIST baseline](https://github.com/adap/flower/tree/main/baselines/fedprox) as an example of a baseline that followed this guide.

> :warning: Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.

**Paper:** https://proceedings.mlr.press/v151/jee-cho22a.html

**Authors:** Yae Jee Cho, Jianyu Wang, Gauri Joshi

**Abstract:** Federated learning is a distributed optimization paradigm that enables a large number of resource-limited client nodes to cooperatively train a model without data sharing. Previous works analyzed the convergence of federated learning by accounting of data heterogeneity, communication/computation limitations, and partial client participation. However, most assume unbiased client participation, where clients are selected such that the aggregated model update is unbiased. In our work, we present the convergence analysis of federated learning with biased client selection and quantify how the bias affects convergence speed. We show that biasing client selection towards clients with higher local loss yields faster error convergence. From this insight, we propose Power-of-Choice, a communication- and computation-efficient client selection framework that flexibly spans the trade-off between convergence speed and solution bias. Extensive experiments demonstrate that Power-of-Choice can converge up to 3 times faster and give **10** higher test accuracy than the baseline random selection.


## About this baseline

****What’s implemented:**** The code in this directory replicates the experiments in *Towards Understanding Biased Client Selection in Federated Learning*(Jee Cho et al., 2022), using MLP on FMNIST and CNN on CIFAR10 for Image Classification. Concretely, it replicates the results for FMNIST in Figure 4 and for CIFAR10 in Figure 6.

****Datasets:**** FMNIST, CIFAR10 from Keras

****Hardware Setup:**** These experiments were run on a desktop machine with 10 CPU threads. Any machine with 4 CPU cores or more would be able to run it in a reasonable amount of time. Note: the entire experiment runs on CPU-only mode.

****Contributors:**** Andrea Restelli (Politecnico di Milano, Italy)


## Experimental Setup

****Task:**** Image classification

****Model:**** This directory implements two models:
* A Multi Layer Perceptron (MLP) used in Power of Choice paper for FMNIST. 
This is the model used by default.
* A CNN used in the paper on CIFAR10 dataset. To use this model you have to set is_cnn=True in the configuration file base.yaml.

****Dataset:**** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

****Training Hyperparameters:**** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

:warning: _The Python environment for all baselines should follow these guidelines in the `EXTENDED_README`. Specify the steps to create and activate your environment. If there are any external system-wide requirements, please include instructions for them too. These instructions should be comprehensive enough so anyone can run them (if non standard, describe them step-by-step)._


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

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
