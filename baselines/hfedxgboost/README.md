---
title: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates
URL:  https://arxiv.org/abs/2304.07537
labels: ["cross-silo", "tree-based", "XGBoost", "Horizontal federated XGBoost", "Classification", "Regression", "Tabular Datasets"] # please add between 4 and 10 single-word (maybe two-words) labels (e.g. "system heterogeneity", "image classification", "asynchronous", "weight sharing", "cross-silo")
dataset: [a9a, cod-rna, ijcnn1, space_ga] # list of datasets you include in your baseline
---

# HFedXgboost: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** [arxiv.org/abs/2304.07537](https://arxiv.org/abs/2304.07537)
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

****What’s implemented:**** The code in this directory replicates the experiments in "Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates" (Ma et al., 2023) for a9a, cod-rna, ijcnn1, space_ga datasets, which proposed the FedXGBllr algorithm. Concretely, it replicates the results for a9a, cod-rna, ijcnn1, space_ga datasets in Table 2.

****Datasets:**** a9a, cod-rna, ijcnn1, space_ga

****Hardware Setup:**** 

****Contributors:**** [Aml Hassan Esmil](https://github.com/Aml-Hassan-Abd-El-hamid)

## Experimental Setup

****Task:**** Tabular classification and regression

****Model:****: XGBoost model combined with 1-layer CNN

****Dataset:**** 
This baseline only includes 7 datasets with a focus on 4 of them (a9a, cod-rna, ijcnn1, space_ga).

Each dataset can be partitioned across 2, 5 or 10 clients in an IID distribution.

| task type  | Dataset | no.of features | no.of samples | 
| :---: | :---: | :---: | :---: |
| Binary classification | a9a<br>cod-rna<br>ijcnn1  | 123<br>8<br>22 | 32,561<br>59,5358<br>49,990 |
| Regression | abalone<br>a9a<br>cpusmall<br>space_ga<br>YearPredictionMSD | 8<br>123<br>12<br>6<br>90 | 4,177<br>32,561<br>8,192<br>3,167<br>515,345 |


****Training Hyperparameters:**** 


## Environment Setup

#### Steps to set up env:
1- Install **pyenv**, follow the instructions from this: https://github.com/pyenv/pyenv-installer 
Note: if you faced the following warning: warning: seems you still have not added 'pyenv' to the load path. and you're not capable of using pyenv in the terminal, you might need to check out this issue: https://github.com/pyenv/pyenv-installer/issues/112
specifically, try the following script:
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

```bash
#to run all the experiments for the centralized model with the original paper config for all the datasets
#gives the output shown in Table 1
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

# The main experiment implemented in your baseline using default hyperparameters (that should be set in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

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


```bash
#results in table 2
python -m hfedxgboost.main --multirun clients="a9a_2_clients","a9a_5_clients","a9a_10_clients" dataset=a9a

#results in table 3
python -m hfedxgboost.main --multirun clients="cod_rna_2_clients","cod_rna_5_clients","cod_rna_10_clients" dataset=cod_rna

#results in table 4
python -m hfedxgboost.main --multirun clients="ijcnn1_2_clients","ijcnn1_5_clients","ijcnn1_10_clients" dataset=ijcnn1

#results in table 5
python -m hfedxgboost.main --multirun clients="space_ga_2_clients","space_ga_5_clients","space_ga_10_clients" dataset=space_ga

#results in table 6
python -m hfedxgboost.main --multirun clients="abalone_2_clients","abalone_5_clients","abalone_10_clients" dataset=abalone

#results in table 7
python -m hfedxgboost.main --multirun clients="cpusmall_2_clients","cpusmall_5_clients","cpusmall_10_clients" dataset=cpusmall

```
### Table 1

| Dataset | task type | test result | 
| :---: | :---: | :---: |
| a9a | Binary classification | .84 |
| cod-rna | Binary classification | .97 |
| ijcnn1 | Binary classification | .98 |
| abalone | Regression | 4.6 |
| cpusmall | Regression | 9 |
| space_ga | Regression | .032 |
| YearPredictionMSD | Regression | 76.41 |

### Those results don't come from following the original paper hyper-parameters, the new hyper-parameters are specified in the config files in the `clients` folder

### Table 2 a9a dataset
|  no.of clients | server-side test Accuracy 
| :---: | :---: |
| 2 | .84
| 5 | .84
| 10 | .83
### Table 3 cod_rna dataset
|  no.of clients | server-side test Accuracy 
| :---: | :---: |
| 2 | .96
| 5 | .96
| 10 | .95
### Table 4 ijcnn1 dataset
|  no.of clients | server-side test Accuracy 
| :---: | :---: |
| 2 | .98
| 5 | .97
| 10 | .96
### Table 5 space_ga dataset
|  no.of clients | server-side test MSE 
| :---: | :---: |
| 2 | .024
| 5 | .033
| 10 | .034
### Table 6 abalone dataset
|  no.of clients | server-side test MSE 
| :---: | :---: |
| 2 | 10
| 5 | 10
| 10 | 10


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
