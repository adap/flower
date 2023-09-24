---
title: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates
url:  https://arxiv.org/abs/2304.07537
labels: ["cross-silo", "tree-based", "XGBoost", "Horizontal federated XGBoost", "Classification", "Regression", "Tabular Datasets"] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [a9a, cod-rna, ijcnn1, abalone, cpusmall, space_ga] # list of datasets you include in your baseline
---

# *hfedxgboost*

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

> :warning: This is the template to follow when creating a new Flower Baseline. Please follow the instructions in `EXTENDED_README.md`

> :warning: Please follow the instructions carefully. You can see the [FedProx-MNIST baseline](https://github.com/adap/flower/tree/main/baselines/fedprox) as an example of a baseline that followed this guide.

> :warning: Please complete the metadata section at the very top of this README. This generates a table at the top of the file that will facilitate indexing baselines.

****Paper:**** :warning: *_add the URL of the paper page (not to the .pdf). For instance if you link a paper on ArXiv, add here the URL to the abstract page (e.g. https://arxiv.org/abs/1512.03385). If your paper is in from a journal or conference proceedings, please follow the same logic._*

****Authors:**** :warning: *_list authors of the paper_*

****Abstract:**** :warning: *_add here the abstract of the paper you are implementing_*


## About this baseline

****What’s implemented:**** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

****Datasets:**** :warning: *_List the datasets you used (if you used a medium to large dataset, >10GB please also include the sizes of the dataset)._*

****Hardware Setup:**** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

****Contributors:**** :warning: *_let the world know who contributed to this baseline. This could be either your name, your name and affiliation at the time, or your GitHub profile name if you prefer. If multiple contributors signed up for this baseline, please list yourself and your colleagues_*


## Experimental Setup

****Task:**** :warning: *_what’s the primary task that is being federated? (e.g. image classification, next-word prediction). If you have experiments for several, please list them_*

****Model:**** :warning: *_provide details about the model you used in your experiments (if more than use a list). If your model is small, describing it as a table would be :100:. Some FL methods do not use an off-the-shelve model (e.g. ResNet18) instead they create your own. If this is your case, please provide a summary here and give pointers to where in the paper (e.g. Appendix B.4) is detailed._*

****Dataset:**** :warning: *_Earlier you listed already the datasets that your baseline uses. Now you should include a breakdown of the details about each of them. Please include information about: how the dataset is partitioned (e.g. LDA with alpha 0.1 as default and all clients have the same number of training examples; or each client gets assigned a different number of samples following a power-law distribution with each client only instances of 2 classes)? if  your dataset is naturally partitioned just state “naturally partitioned”; how many partitions there are (i.e. how many clients)? Please include this an all information relevant about the dataset and its partitioning into a table._*

****Training Hyperparameters:**** :warning: *_Include a table with all the main hyperparameters in your baseline. Please show them with their default value._*


## Environment Setup

#### Steps to set up env:
1- Install **pyenv**, follow the instructions from this: https://github.com/pyenv/pyenv-installer 
Note: if you faced the following warning: warning: seems you still have not added 'pyenv' to the load path. and you're not capable of using pyenv in the terminal, you might need to check out this issue: https://github.com/pyenv/pyenv-installer/issues/112
specifically try the following script:
```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init --path)"
eval "$(pyenv virtualenv-init -)"
```
2- Install **poetry** at the system level: https://python-poetry.org/docs/ --> running curl -sSL https://install.python-poetry.org | python3 -(dont' forget to add to your .bashrc or equivalent)

3- Then install a version of Python of your choice via pyenv, eg: pyenv install 3.10.6

4- In the Horizontal_XGBoost directory where you can see pyproject.toml, write the following commands in your terminal:
```
pyenv local 3.10.6
poetry env use 3.10.6
poetry install 
```

Run `Poetry shell` in your terminal to activate the environment. 
## Running the Experiments

:warning: _Provide instructions on the steps to follow to run all the experiments._
```bash
#to run all the experiments for the centralized model
python -m hfedxgboost.main --config-name "Centralized_Baseline"

#to run the federated version for a9a dataset with 2 clients
python -m hfedxgboost.main dataset="a9a" clients="a9a_2_clients"

#to run the federated version for a9a dataset with 5 clients
python -m hfedxgboost.main dataset="a9a" clients="a9a_5_clients"

#to run the federated version for a9a dataset with 10 clients
python -m hfedxgboost.main dataset="a9a" clients="a9a_10_clients"

#to run the federated version for cod-rna dataset with 2 clients
python -m hfedxgboost.main dataset="cod_rna" clients="cod_rna_2_clients"

#to run the federated version for cod-rna dataset with 5 clients
python -m hfedxgboost.main dataset="cod_rna" clients="cod_rna_5_clients"

#to run the federated version for cod-rna dataset with 10 clients
python -m hfedxgboost.main dataset="cod_rna" clients="cod_rna_10_clients"

#to run the federated version for space_ga dataset with 2 clients
python -m hfedxgboost.main dataset="space_ga" clients="space_ga_2_clients"


#to run the federated version for space_ga dataset with 5 clients
python -m hfedxgboost.main dataset="space_ga" clients="space_ga_5_clients"

#to run the federated version for space_ga dataset with 10 clients
python -m hfedxgboost.main dataset="space_ga" clients="space_ga_10_clients"

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
