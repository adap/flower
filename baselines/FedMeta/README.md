---
title: Federated Meta-Learning with Fast Convergence and Efficient Communication
url: https://arxiv.org/abs/1802.07876
labels: [meta learning, maml, meta-sgd, personalization] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [Femnist, Shakespeare] # list of datasets you include in your baseline
---

# FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication

****Paper:**** : https://arxiv.org/abs/1802.07876

****Authors:**** :Fei Chen, Mi Luo, Zhenhua Dong, Zhenguo Li, Xiuqiang He

****Abstract:**** :Statistical and systematic challenges in collaboratively training machine learning models across distributed networks of mobile devices have been the bottlenecks in the real-world application of federated learning. In this work, we show that meta-learning is a natural choice to handle these issues, and propose a federated meta-learning framework FedMeta, where a parameterized algorithm (or meta-learner) is shared, instead of a global model in previous approaches. We conduct an extensive empirical evaluation on LEAF datasets and a real-world production dataset, and demonstrate that FedMeta achieves a reduction in required communication cost by 2.82-4.33 times with faster convergence, and an increase in accuracy by 3.23%-14.84% as compared to Federated Averaging (FedAvg) which is a leading optimization algorithm in federated learning. Moreover, FedMeta preserves user privacy since only the parameterized algorithm is transmitted between mobile devices and central servers, and no raw data is collected onto the servers.


## About this baseline

****What’s implemented:**** : I implement ~~

****Datasets:**** :

****Hardware Setup:**** :

****Contributors:**** :


## Experimental Setup

****Task:**** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

****Model:**** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

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
