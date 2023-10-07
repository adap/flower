---
title: Federated Self-training for Semi-supervised Audio Recognition
url: [dl.acm.org/doi/10.1145/3520128](https://dl.acm.org/doi/10.1145/3520128)
labels: [Audio Classification, Semi Supervised learning]
dataset: [Ambient Context, Speech Commands]
---

# FedStar: Federated Self-training for Semi-supervised Audio Recognition

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** :[dl.acm.org/doi/10.1145/3520128](https://dl.acm.org/doi/10.1145/3520128)

****Authors:**** : Vasileios Tsouvalas, Aaqib Saeed, Tanir Özcelebi

****Abstract:**** : Federated Learning is a distributed machine learning paradigm dealing with decentralized and personal datasets. Since data reside on devices such as smartphones and virtual assistants, labeling is entrusted to the clients or labels are extracted in an automated way. Specifically, in the case of audio data, acquiring semantic annotations can be prohibitively expensive and time-consuming. As a result, an abundance of audio data remains unlabeled and unexploited on users’ devices. Most existing federated learning approaches focus on supervised learning without harnessing the unlabeled data. In this work, we study the problem of semi-supervised learning of audio models via self-training in conjunction with federated learning. We propose FedSTAR to exploit large-scale on-device unlabeled data to improve the generalization of audio recognition models. We further demonstrate that self-supervised pre-trained models can accelerate the training of on-device models, significantly improving convergence within fewer training rounds. We conduct experiments on diverse public audio classification datasets and investigate the performance of our models under varying percentages of labeled and unlabeled data. Notably, we show that with as little as 3% labeled data available, FedSTAR on average can improve the recognition rate by 13.28% compared to the fully supervised federated model.


## About this baseline

****What’s implemented:**** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

****Datasets:**** : Ambient Context, Speech Commands

****Hardware Setup:**** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

****Contributors:**** Raj Parekh [GitHub](https://github.com/Raj-Parekh24), [Mail](rajparekh144@gmail.com)


## Environment Setup
# Set python version
pyenv local 3.10.6
# Tell poetry to use python 3.10
poetry env use 3.10.6
# Now install the environment
poetry install
# Start the shell to activate your environment.
poetry shell
# Run dataset shell script
. setup_datasets.sh

## Running the Experiments
###### python -m fedstar.server
###### python -m fedstar.client

## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run python -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
