---
title: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates
URL:  https://arxiv.org/abs/2304.07537
labels: ["cross-silo", "tree-based", "XGBoost", "Horizontal federated XGBoost", "Classification", "Regression", "Tabular Datasets"] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [a9a, cod-rna, ijcnn1, abalone, cpusmall, space_ga] # list of datasets you include in your baseline
---

# HFedXgboost: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** https://arxiv.org/abs/2304.07537
****Authors:**** Chenyang Ma, Xinchi Qiu, Daniel J. Beutel, Nicholas D. Lane

****Abstract:**** The privacy-sensitive nature of decentralized datasets and
the robustness of eXtreme Gradient Boosting (XGBoost) on
tabular data raise the need to train XGBoost in the con-
text of federated learning (FL). Existing works on federated
XGBoost in the horizontal setting rely on the sharing of gradients, which induce per-node level communication frequency
and serious privacy concerns. To alleviate these problems, we
develop an innovative framework for horizontal federated
XGBoost which does not depend on the sharing of gradients and simultaneously boosts privacy and communication
efficiency by making the learning rates of the aggregated
tree ensembles are learnable. We conduct extensive evaluations
on various classification and regression datasets, showing
our approach achieves performance comparable to the state-of-the-art method and effectively improves communication
efficiency by lowering both communication rounds and communication overhead by factors ranging from 25x to 700x.


## About this baseline

****What’s implemented:**** :warning: *_Concisely describe what experiment(s) in the publication can be replicated by running the code. Please only use a few sentences. Start with: “The code in this directory …”_*

****Datasets:**** a9a, cod-rna, ijcnn1, space_ga

****Hardware Setup:**** :warning: *_Give some details about the hardware (e.g. a server with 8x V100 32GB and 256GB of RAM) you used to run the experiments for this baseline. Someone out there might not have access to the same resources you have so, could list the absolute minimum hardware needed to run the experiment in a reasonable amount of time ? (e.g. minimum is 1x 16GB GPU otherwise a client model can’t be trained with a sufficiently large batch size). Could you test this works too?_*

****Contributors:**** [Aml Hassan Esmil](https://github.com/Aml-Hassan-Abd-El-hamid)

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
#to run all the experiments for the centralized model with the original paper config for all the datasets
python -m hfedxgboost.main --config-name "centralized_basline_all_datasets_paper_config"

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
## How to add a new dataset

This code doesn't cover all the datasets from the paper yet, so if you wish to add a new dataset, here are the steps:

**1- you need to download the dataset from its source:**
- In the `dataset_preparation.py` file, specifically in the `download_data` function add the code to download your dataset -or if you already downloaded it manually add the code to return its file path- it could look something like the following example:
```
if dataset_name=="<the name of your dataset>":
        DATASET_PATH=os.path.join(ALL_DATASETS_PATH, "<the name of your dataset>")
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
            urllib.request.urlretrieve(
                "<the URL which from the first file of your dataset will be downloaded>",
                f"{os.path.join(DATASET_PATH, '<the name of your dataset's first file>')}",
            )
            urllib.request.urlretrieve(
                "<the URL which from the second file of your dataset will be downloaded>",
                f"{os.path.join(DATASET_PATH, '<the name of your dataset's second file>')}",
            )
        # if the 2 files of your dataset are divided into training and test file put the training then test ✅
        return [os.path.join(DATASET_PATH, '<the name of your dataset's first file>'),os.path.join(DATASET_PATH, '<the name of your dataset's second file>')]  
```
that function will be called in the `dataset.py` file in the `load_single_dataset` function and the different files of your dataset will be concatenated -if your dataset is one file then nothing will happen it will just be loaded- using the `datafiles_fusion` function from the `dataset_preparation.py` file. 

:warning: if any of your dataset's files end with `.bz2` you have to add the following piece of code before the return line and inside the `if` condition
```
for filepath in os.listdir(DATASET_PATH):
                abs_filepath = os.path.join(DATASET_PATH, filepath)
                with bz2.BZ2File(abs_filepath) as fr, open(abs_filepath[:-4], "wb") as fw:
                    shutil.copyfileobj(fr, fw)
```

:warning: `datafiles_fusion` function uses `sklearn.datasets.load_svmlight_file` to load the dataset, if your dataset is `csv` or something that function won't work on it and you will have to alter the `datafiles_fusion` function to work with you dataset files format.

**2-Add config files for your dataset:**

**a- config files for the centralized baseline:**

- To run the centralized model on your dataset with the original hyper-parameters from the paper alongside all the other datasets added before just do the following step:
   - in the dictionary called `dataset_tasks` in the `utils.py` file add your dataset name as a key -the same name that you put in the `download_data` function  in the step before- and add its task type, this code perform for 2 tasks: `BINARY` which is binary classification or `REG` which is regression.
    
- To run the centralized model on your dataset you need to create a config file `<your dataset>.yaml` in the `xgboost_params_centralized` folder and another .yaml file in the `dataset` folder -you will find that one of course inside the `conf` folder :) - and you need to specify the hyper-parameters of your choice for the xgboost model

  - the .yaml file in the `dataset` folder should look something like this:
    ```
    defaults:
       - task: <config file name with your task type, "Regression" or "Binary_Classification">
    dataset_name: "<your dataset name, must be the same name used in the download_data function>"
    train_ratio: <percentage of the training data of the whole dataset>
    ```
  - the .yaml file in the `xgboost_params_centralized` folder should contain the values for all the hyper-parameters of your choice for the xgboost model

**b- config files for the federated baseline:**

To run the federated baseline with your dataset, you need to first create the .yaml file in the `dataset` folder that was mentioned before and you need to create config files that contain the no.of the clients and it should look something like this:
```
n_estimators_client: <int used in xgboost and CNN model>
num_rounds: <int to indicate the round of communications between the server and the clients>
client_num: <int to indicate the number of clients>
num_iterations: <int to set the no.of iteration for the CNN model>
xgb:
  max_depth: <int>
CNN:
  lr: <float>
```
