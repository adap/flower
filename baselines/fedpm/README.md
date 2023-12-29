---
title: Sparse Random Networks for Communication-Efficient Federated Learning
url: https://openreview.net/forum?id=k1FHgri5y3-
labels: [Communication-efficiency, mask training, compression]
dataset: MNIST, CIFAR-10, CIFAR-100]
---

# Sparse Random Networks for Communication-Efficient Federated Learning

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [openreview.net/forum?id=k1FHgri5y3-](https://openreview.net/forum?id=k1FHgri5y3-)

**Authors:** Berivan Isik, Francesco Pase, Deniz Gunduz, Tsachy Weissman, Zorzi Michele

**Abstract:** One main challenge in federated learning is the large communication cost of exchanging weight updates from clients to the server at each round. While prior work has made great progress in compressing the weight updates through gradient compression methods, we propose a radically different approach that does not update the weights at all. Instead, our method freezes the weights at their initial \emph{random} values and learns how to sparsify the random network for the best performance. To this end, the clients collaborate in training a \emph{stochastic} binary mask to find the optimal sparse random network within the original one. At the end of the training, the final model is a sparse network with random weights -- or a subnetwork inside the dense random network. We show improvements in accuracy, communication (less than 1 bit per parameter (bpp)), convergence speed, and final model size (less than 1 bpp) over relevant baselines on MNIST, EMNIST, CIFAR-10, and CIFAR-100 datasets, in the low bitrate regime.


## About this baseline

**What’s implemented:** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

**Datasets:** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset)._*

**Hardware Setup:** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

**Contributors:** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

**Task:** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

**Model:** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

**Dataset:** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

**Training Hyperparameters:** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

To run this baseline both [`pyenv`](https://github.com/pyenv/pyenv) and [`poetry`](https://python-poetry.org/docs/) are assumed to be already present in your system. Then, follow these steps:

```bash
# Set directory to use python 3.10 (install with `pyenv install <version>` if you don't have it)
pyenv local 3.10.12

# Tell poetry to use python3.10
poetry env use 3.10.12

# Install all dependencies
poetry install

# Activate your environment
poetry shell

```

## Running the Experiments

:warning: _Provide instructions on the steps to follow to run all the experiments._
```bash  
# Run with default arguments
python -m fedpm.main
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
