---
title: A Byzantine-Resilient Aggregation Scheme for Federated Learning via Matrix Autoregression on Client Updates
url: https://arxiv.org/abs/2303.16668
labels: [robustness, cross-silo, model poisoning, anomaly detection, autoregressive model] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [MNIST, Income, California Housing] # list of datasets you include in your baseline
---

# :warning:*_Title of your baseline_*

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

> :warning: This is the template to follow when creating a new Flower Baseline. Please follow the instructions in `EXTENDED_README.md`

> :warning: Please follow the instructions carefully. You can see the [FedProx-MNIST baseline](https://github.com/adap/flower/tree/main/baselines/fedprox) as an example of a baseline that followed this guide.

> :warning: Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.

****Paper:**** https://arxiv.org/abs/2303.16668

****Authors:**** Gabriele Tolomei, Edoardo Gabrielli, Dimitri Belli, Vittorio Miori

****Abstract:**** In this work, we propose FLANDERS, a novel federated learning (FL) aggregation scheme robust to Byzantine attacks. FLANDERS considers the local model updates sent by clients at each FL round as a matrix-valued time series. Then, it identifies malicious clients as outliers of this time series by comparing actual observations with those estimated by a matrix autoregressive forecasting model. Experiments conducted on several datasets under different FL settings demonstrate that FLANDERS matches the robustness of the most powerful baselines against Byzantine clients. Furthermore, FLANDERS remains highly effective even under extremely severe attack scenarios, as opposed to existing defense strategies. 


## About this baseline

****What’s implemented:**** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

****Datasets:**** MNIST, Income, California Housing

****Hardware Setup:**** Apple M2 Pro, 16gb RAM

****Contributors:**** Edoardo Gabrielli, University of Rome "La Sapienza"


## Experimental Setup

****Task:**** Image classification, logistic regression, linear regression

****Models:**** Appendix C of the paper describe the models, but here's a summary.

Income (binary classification):
- cyclic coordinate descent (CCD)
- L1-regularized binary cross-entropy loss (LASSO)

MNIST (multilabel classification, fully connected, feed forward NN):
- Multilevel Perceptron (MLP)
- minimizing multiclass cross-entropy loss using Adam optimizer
- input: 784
- hidden layer 1: 128
- hidden layer 2: 256

California Housing (linear regression):
- cyclic coordinate descent (CCD)
- L1/L2-regularized mean squared error (LASSO/RIDGE)


****Dataset:**** Every dataset is partitioned into two disjoint sets: 80% for training and 20% for testing. The training set is distributed uniformly across all clients (100), while the testing set is held by the server to evaluate the global model.

| Description | Default Value |
| ----------- | ----- |
| Partitions | 100 |
| Evaluation | centralized |
| Training set | 80% |
| Testing set | 20% |

****Training Hyperparameters:****

| Dataset | # of clients  | Clients per round | # of rounds | Batch size | Learning rate | $\lambda_1$ | $\lambda_2$ | Optimizer | Dropout | Alpha | Beta | # of clients to keep | Sampling |
| -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- | -- |
| Income | 100 | 100 | 50 | \ | \ | 1.0 | 0.0 | CCD | \ | 0.0 | 0.0 | 1 | \ |
| MNIST | 100 | 100 | 50 | 32 | $10^{-3}$ | \ | \ | Adam | 0.2 | 0.0 | 0.0 | 1 | \ |
| California Housing | 100 | 100 | 50 | \ | \ | 0.5 | 0.5 | CCD | \ | 0.0 | 0.0 | 1 | \ |

Where $\lambda_1$ and $\lambda_2$ are Lasso and Ridge regularization terms.

TODO: Might add CIFAR-10.

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
