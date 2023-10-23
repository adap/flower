---
title: HETEROFL- COMPUTATION AND COMMUNICATION EFFICIENT FEDERATED LEARNING FOR HETEROGENEOUS CLIENTS
url: https://openreview.net/forum?id=TNkPBBYFkXg
labels: [system heterogeneity, image classification]
dataset: [MNIST, CIFAR10]
---

# *HeteroFL*

****Paper:**** [openreview.net/forum?id=TNkPBBYFkXg](https://openreview.net/forum?id=TNkPBBYFkXg)

****Authors:**** *Enmao Diao, Jie Ding, Vahid Tarokh*

****Abstract:**** *Federated Learning (FL) is a method of training machine learning models on private data distributed over a large number of possibly heterogeneous clients such as mobile phones and IoT devices. In this work, we propose a new federated learning framework named HeteroFL to address heterogeneous clients equipped with very different computation and communication capabilities. Our solution can enable the training of heterogeneous local models with varying computation complexities and still produce a single global inference model. For the first time, our method challenges the underlying assumption of existing work that local models have to share the same architecture as the global model. We demonstrate several strategies to enhance FL training and conduct extensive empirical evaluations, including five computation complexity levels of three model architecture on three datasets. We show that adaptively distributing subnetworks according to clients’ capabilities is both computation and communication efficient.*


## About this baseline

****What’s implemented:****  *The code in this directory is an implementation of HeteroFL in pytorch using flower. The code incorporates references from the authors' implementation. I have also implemented custom model split and aggregation as suggested by @negedng, which is available here(github.com/msck72/heterofl_custom_aggregation). By modifying the configuration in the base.yaml, the results in the paper can be replicated, with both fixed and dynamic computational complexities among clients.*

*Model rate defines the computational complextiy of a client. Authors have defined five different computation complexity levels {a, b, c, d, e} with the hidden channel shrinkage ratio r = 0.5.
Model split mode specifies whether the computaional complexities of clients are fixed (throughout the experiment), or whether they are dynamic (change their mode_rate/computational-complexity every-round).
Model mode determines the proportionality of clients with various computation complexity levels, for example, a4-b2-e4 determines at each round, proportion of clients with computational complexity level a = 4 / (4 + 2 + 4) * num_clients , similarly, proportion of clients with computational complexity level b = 2 / (4 + 2 + 4) * num_clients and so on.*

*ModelRateManager manages the model rate of client in simulation, which changes the model rate based on the model mode of the setup and ClientManagerHeterofl keeps track of model rates of the clients, so that configure fit knows which/how-much subset of the model that needs to be sent to the client.*

****Datasets:**** *The code utilized benchmark datasets such as MNIST and CIFAR-10 for its experimentation.*

****Hardware Setup:****  *The experiments were run on Google colab pro with 50GB RAM and T4 TPU.*

****Contributors:**** *M S Chaitanya Kumar [(github.com/msck72)](github.com/msck72)*


## Experimental Setup

****Task:**** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

****Model:**** *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

****Dataset:**** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

****Training Hyperparameters:**** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

```
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

```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

poetry run -m heterofl.main  # Which runs the heterofl with arguments availbale in heterfl/conf/base.yaml

# We could override the settings that were specified in base.yaml using the command-line-arguments
# Here's an example for changing the dataset name, non-iid and model
poetry run -m heterofl.main dataset.dataset_name='CIFAR10' dataset.iid=False model.model_name='resnet18'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
poetry run -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

# Similarly, another example for changing num_rounds, model_split_mode, and model_mode
poetry run -m heterofl.main num_rounds=400 control.model_split_mode='dynamic' control.model_mode='a1-b1'

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
