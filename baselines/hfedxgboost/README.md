---
title: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates
URL:  https://arxiv.org/abs/2304.07537
labels: ["cross-silo", "tree-based", "XGBoost", "Horizontal federated XGBoost", "Classification", "Regression", "Tabular Datasets"] 
dataset: [a9a, cod-rna, ijcnn1, space_ga, cpusmall, YearPredictionMSD] 
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
| Regression | abalone<br>cpusmall<br>space_ga<br>YearPredictionMSD | 8<br>12<br>6<br>90 | 4,177<br>8,192<br>3,167<br>515,345 |


****Training Hyperparameters:**** 

Using the original hyperparameters didn't give the best performance sometimes, It's kinda normal as the paper used the same hyperparameters for all the datasets while some of them were small with few no.of features while some were big datasets with bigger no.of features, e.g: YearPredictionMSD had 515,345 rows and 90 features while space_ga got 3,167 rows and 6 features.<br>
There are different hyperparameters used for each client setting in the federated system.<br>
For the centralized model, I mostly used the paper's hyperparameters as they give very good results -except for abalone and cpusmall-, here are the used hyperparameters:

| Hyperparameter name | value |
| -- | -- |
| n_estimators | 500 |
| max_depth | 8 |
| subsample | 0.8 |
| learning_rate | .1 |
| colsample_bylevel | 1 |
| colsample_bynode | 1 |
| colsample_bytree | 1 |
| alpha | 5 |
| gamma | 5 |
| num_parallel_tree | 1 |
| min_child_weight | 1 |

To help with the fine-tuning of the hyperparameters process, there are 2 classes in the utils.py that write down the used hyperparameters in the experiments and the results for that experiment in 2 separate CSV files, some of the hyperparameters used in the experiments done during building this baseline can be found in results.csv and results_centralized.csv files.

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

#to run the experiments for the centralized model  with customized hyperparameters
#write a command that looks like that in your terminal
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=<name of the config file that contain the dataset name, task,...> xgboost_params_centralized=<name of the config file containing the xgboot customized hyperparameters>
#e.g
#to run the centralized model with customized hyperparameters for cpusmall dataset, it should give 7 test MSE which is better than the 9 MSE that #the default centralized model hyperparameters give.
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=cpusmall xgboost_params_centralized=cpusmall_xgboost_centralized

#to run the federated version for any dataset with no.of clients
python -m hfedxgboost.main dataset=<name of the config file that contain the dataset name, task,...> clients=<name of the config file containing the customized hyperparameters and the no.of clients>
#e.g
#to run the federated version for a9a dataset with 5 clients
python -m hfedxgboost.main dataset=a9a clients=a9a_5_clients

#if you wish to change any parameters from any config file from the terminal, then you should follow this formula
python -m hfedxgboost.main folder=config_file_name folder.parameter_name=its new value
#e.g:
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=abalone xgboost_params_centralized=abalone_xgboost_centralized xgboost_params_centralized.max_depth=8 dataset.train_ratio=.80
```


## Expected Results


```bash
#to run all the experiments for the centralized model with the original paper config for all the datasets
#gives the output shown in Table 1
python -m hfedxgboost.main --config-name "centralized_basline_all_datasets_paper_config"

#results for a9a dataset in table 2 
python -m hfedxgboost.main --multirun clients="a9a_2_clients","a9a_5_clients","a9a_10_clients" dataset=a9a

#results for cod_rna dataset in table 2
python -m hfedxgboost.main --multirun clients="cod_rna_2_clients","cod_rna_5_clients","cod_rna_10_clients" dataset=cod_rna

#results for ijcnn1 dataset in table 2
python -m hfedxgboost.main --multirun clients="ijcnn1_2_clients","ijcnn1_5_clients","ijcnn1_10_clients" dataset=ijcnn1

#results for space_ga dataset in table 3
python -m hfedxgboost.main --multirun clients="space_ga_2_clients","space_ga_5_clients","space_ga_10_clients" dataset=space_ga

#results for abalone dataset in table 3
python -m hfedxgboost.main --multirun clients="abalone_2_clients","abalone_5_clients","abalone_10_clients" dataset=abalone

#results for cpusmall dataset in table 3
python -m hfedxgboost.main --multirun clients="cpusmall_2_clients","cpusmall_5_clients","cpusmall_10_clients" dataset=cpusmall

#results for YearPredictionMSD_2 dataset in table 3
python -m hfedxgboost.main --multirun clients=YearPredictionMSD_2_clients,YearPredictionMSD_5_clients,YearPredictionMSD_10_clients dataset=YearPredictionMSD
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
 
**Those results don't come from following the original paper hyper-parameters, the new hyper-parameters are specified in the config files in the `clients` folder**

### Table 2

| Dataset | task type |no. of clients | server-side test Accuracy |
| :---: | :---: | :---: | :---: |
| a9a | Binary Classification | 2<br>5<br>10 | 0.84<br>0.84<br>0.83 |
| cod_rna | Binary Classification | 2<br>5<br>10 | 0.96<br>0.96<br>0.95 | 
| ijcnn1 | Binary Classification |2<br>5<br>10 | 0.98<br>0.97<br>0.96 |

### Table 3

| Dataset | task type |no. of clients | server-side test MSE |
| :---: | :---: | :---: | :---: |
| space_ga | Regression | 2<br>5<br>10 | 0.024<br>0.033<br>0.034 |
| abalone | Regression | 2<br>5<br>10 | 10<br>10<br>10 | 
| YearPredictionMSD | Regression | 2<br>5<br>10 | 119<br>118<br>118 | 


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
