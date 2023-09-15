---
title: Personalized Federated Learning using Hypernetworks
url: https://arxiv.org/abs/2103.04628
labels: ["data heterogenity", "hypernetworks","personalised federated learning",] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [MNIST, CIFAR10, CIFAR100] # list of datasets you include in your baseline
---

# *pFedHN*



****Paper:**** :*_https://arxiv.org/abs/2103.04628_*

****Authors:**** :*_Aviv Shamsian,Aviv Navon,Ethan Fetaya,Gal Chechik_*

****Abstract:**** :*_Personalized federated learning is tasked with training machine learning models for multiple clients, each with its own data distribution. The goal is to train personalized models in a collaborative way while accounting for data disparities across clients and reducing communication costs. We propose a novel approach to this problem using hypernetworks, termed pFedHN for personalized Federated HyperNetworks. In this approach, a central hypernetwork model is trained to generate a set of models, one model for each client. This architecture provides effective parameter sharing across clients, while maintaining the capacity to generate unique and diverse personal models. Furthermore, since hypernetwork parameters are never transmitted, this approach decouples the communication cost from the trainable model size. We test pFedHN empirically in several personalized federated learning challenges and find that it outperforms previous methods. Finally, since hypernetworks share information across clients we show that pFedHN can generalize better to new clients whose distributions differ from any client observed during training._*

## About this baseline

****What’s implemented:**** :*_"The code in the repository reproduces the paper by implementing the concept of hypernetworks which create weights for that target network thus resolving the problems of dataheterogenity.The hypernetworks lies in the server and the clients have the target model. Initially hypernetwork sends the weights from the server which is loaded into the target net . After the targetnet is trained we pass the delta_theta to the client , which inturn updated the phi_gradients for the hypernetwork and the flow goes on._*

****Datasets:**** :*_[MNIST,CIFAR10,CIFAR100]_*

****Hardware Setup:**** : *_Will be updated_*

****Contributors:**** :*_Ram Samarth B B(@achiverram28) , Kishan Gurumurthy(@kishan-droid) , Sachin DN(@sachugowda)_*


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

poetry run python -m <baseline-name>.main <no additional arguments> # where <baseline-name> is the name of this directory and that of the only sub-directory in this directory (i.e. where all your source code is)

# If you are using a dataset that requires a complicated download (i.e. not using one natively supported by TF/PyTorch) + preprocessing logic, you might want to tell people to run one script first that will do all that. Please ensure the download + preprocessing can be configured to suit (at least!) a different download directory (and use as default the current directory). The expected command to run to do this is:

poetry run python -m <baseline-name>.dataset_preparation <optional arguments, but default should always run>

# It is expected that you baseline supports more than one dataset and different FL settings (e.g. different number of clients, dataset partitioning methods, etc). Please provide a list of commands showing how these experiments are run. Include also a short explanation of what each one does. Here it is expected you'll be using the Hydra syntax to override the default config.

poetry run python -m <baseline-name>.main  <override_some_hyperparameters>
.
.
.
poetry run python -m <baseline-name>.main  <override_some_hyperparameters>
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run python -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
