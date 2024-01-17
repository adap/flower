---
title: Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates
URL:  https://arxiv.org/abs/2304.07537
labels: [cross-silo, tree-based, XGBoost, Classification, Regression, Tabular] 
dataset: [a9a, cod-rna, ijcnn1, space_ga, cpusmall, YearPredictionMSD] 
---

# Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

**Paper:** [arxiv.org/abs/2304.07537](https://arxiv.org/abs/2304.07537)

**Authors:** Chenyang Ma, Xinchi Qiu, Daniel J. Beutel, Nicholas D. Lane

**Abstract:** The privacy-sensitive nature of decentralized datasets and the robustness of eXtreme Gradient Boosting (XGBoost) on tabular data raise the need to train XGBoost in the context of federated learning (FL). Existing works on federated XGBoost in the horizontal setting rely on the sharing of gradients, which induce per-node level communication frequency and serious privacy concerns. To alleviate these problems, we develop an innovative framework for horizontal federated XGBoost which does not depend on the sharing of gradients and simultaneously boosts privacy and communication efficiency by making the learning rates of the aggregated tree ensembles are learnable. We conduct extensive evaluations on various classification and regression datasets, showing our approach achieve performance comparable to the state-of-the-art method and effectively improves communication efficiency by lowering both communication rounds and communication overhead by factors ranging from 25x to 700x.


## About this baseline

**What’s implemented:** The code in this directory replicates the experiments in "Gradient-less Federated Gradient Boosting Trees with Learnable Learning Rates" (Ma et al., 2023) for a9a, cod-rna, ijcnn1, space_ga datasets, which proposed the FedXGBllr algorithm. Concretely, it replicates the results for a9a, cod-rna, ijcnn1, space_ga datasets in Table 2.

**Datasets:** a9a, cod-rna, ijcnn1, space_ga

**Hardware Setup:** Most of the experiments were done on a machine with an Intel® Core™ i7-6820HQ Processor, that processor got 4 cores and 8 threads. 

**Contributors:** [Aml Hassan Esmil](https://github.com/Aml-Hassan-Abd-El-hamid)

## Experimental Setup

**Task:** Tabular classification and regression

**Model:** XGBoost model combined with 1-layer CNN

**Dataset:** 
This baseline only includes 7 datasets with a focus on 4 of them (a9a, cod-rna, ijcnn1, space_ga).

Each dataset can be partitioned across 2, 5 or 10 clients in an IID distribution.

| task type  | Dataset | no.of features | no.of samples | 
| :---: | :---: | :---: | :---: |
| Binary classification | a9a<br>cod-rna<br>ijcnn1  | 123<br>8<br>22 | 32,561<br>59,5358<br>49,990 |
| Regression | abalone<br>cpusmall<br>space_ga<br>YearPredictionMSD | 8<br>12<br>6<br>90 | 4,177<br>8,192<br>3,167<br>515,345 |


**Training Hyperparameters:** 
For the centralized model, the paper's hyperparameters were mostly used as they give very good results -except for abalone and cpusmall-, here are the used hyperparameters -they can all be found in the `yaml` file named `paper_xgboost_centralized`:

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

Here are all the original hyperparameters for the federated horizontal XGBoost model -hyperparameters that are used only in the XGBoost model are initialized with xgb same for the ones only used in Adam-:

| Hyperparameter name | value |
| -- | -- |
| n_estimators | 500/no.of clients |
| xgb max_depth | 8 |
| xgb subsample | 0.8 |
| xgb learning_rate | .1 |
| xgb colsample_bylevel | 1 |
| xgb colsample_bynode | 1 |
| xgb colsample_bytree | 1 |
| xgb alpha | 5 |
| xgb gamma | 5 |
| xgb num_parallel_tree | 1 |
| xgb min_child_weight | 1 |
| Adam learning rate | .0001 |
| Adam Betas | 0.5, 0.999 |
| no.of iterations for the CNN model | 100 |

Those hyperparameters did  well for most datasets but for some datasets, it wasn't giving the best performance so a fine-tuning journey has started in order to achieve better results.<br>
At first, it was a manual process basically experiencing different values for some groups of hyperparameters to explore those hyperparameters's effect on the performance of different datasets until I decided to focus on those groups of the following hyperparameters as they seemed to have the major effect on different datasets performances:
| Hyperparameter name |
| -- | 
| n_estimators |
| xgb max_depth | 
| Adam learning rate | 
| no.of iterations for the CNN model |

All the final new values for those hyperparameters can be found in 3 `yaml` files named `dataset_name_<no. of clients>_clients` and all the original values for those hyperparameters can be found in 3 `yaml` files named `paper_<no. of clients>_clients`. This resulted in a large number of config files 3*7+3= 24 config files in the `clients` folder.

## Environment Setup

These steps assume you have already installed `Poetry` and `pyenv`. In the this directory (i.e. `/baselines/hfedxgboost`) where you can see `pyproject.toml`, execute the following commands in your terminal:

```bash
# Set python version
pyenv local 3.10.6
# Tell Poetry to use it
poetry env use 3.10.6
# Install all dependencies
poetry install 
# Activate your environment
poetry shell
```

## Running the Experiments

With your environment activated you can run the experiments directly. The datasets will be downloaded automatically.

```bash
# to run the experiments for the centralized model with customized hyperparameters run
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=<name-of-config-file> xgboost_params_centralized=<name-of-config-file-with-hyperparameters>
#e.g
# to run the centralized model with customized hyperparameters for cpusmall dataset
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=cpusmall xgboost_params_centralized=cpusmall_xgboost_centralized

# to run the federated version for any dataset with no.of clients
python -m hfedxgboost.main dataset=<name-of-config-file> clients=<name-of-client-config-file>
# for example
# to run the federated version for a9a dataset with 5 clients
python -m hfedxgboost.main dataset=a9a clients=a9a_5_clients

# if you wish to change any parameters from any config file from the terminal, then you should follow this formula
python -m hfedxgboost.main folder=config_file_name folder.parameter_name=its_new_value
#e.g:
python -m hfedxgboost.main --config-name Centralized_Baseline dataset=abalone xgboost_params_centralized=abalone_xgboost_centralized xgboost_params_centralized.max_depth=8 dataset.train_ratio=.80
```


## Expected Results

This section shows how to reproduce some of the results in the paper. Tables 2 and 3 were obtained using different hyperparameters than those indicated in the paper. Without these some experimetn exhibited worse performance. Still, some results remain far from those in the original paper.

### Table 1: Centralized Evaluation
```bash
# to run all the experiments for the centralized model with the original paper config for all the datasets
# gives the output shown in Table 1
python -m hfedxgboost.main --config-name centralized_basline_all_datasets_paper_config

# Please note that unlike in the federated experiments, the results will be only printed on the terminal
# and won't be logged into a file.
```
| Dataset | task type | test result | 
| :---: | :---: | :---: |
| a9a | Binary classification | 84.9% |
| cod-rna | Binary classification | 97.3% |
| ijcnn1 | Binary classification | 98.7% |
| abalone | Regression | 4.6 |
| cpusmall | Regression | 9 |
| space_ga | Regression | .032 |
| YearPredictionMSD | Regression | 76.41 |

### Table 2: Federated Binary Classification

```bash
# Results for a9a dataset in table 2 
python -m hfedxgboost.main --multirun clients=a9a_2_clients,a9a_5_clients,a9a_10_clients dataset=a9a

# Results for cod_rna dataset in table 2
python -m hfedxgboost.main --multirun clients=cod_rna_2_clients,cod_rna_5_clients,cod_rna_10_clients dataset=cod_rna

# Results for ijcnn1 dataset in table 2
python -m hfedxgboost.main --multirun clients=ijcnn1_2_clients,ijcnn1_5_clients,ijcnn1_10_clients dataset=ijcnn1
```

| Dataset | task type |no. of clients | server-side test Accuracy |
| :---: | :---: | :---: | :---: |
| a9a | Binary Classification | 2<br>5<br>10 | 84.4% <br>84.2% <br> 83.7% |
| cod_rna | Binary Classification | 2<br>5<br>10 | 96.4% <br>96.2% <br>95.0%  | 
| ijcnn1 | Binary Classification |2<br>5<br>10 | 98.0% <br>97.28% <br>96.8%  |


### Table 3: Federated Regression
```bash
# Notice that: the MSE results shown in the tables usually happen in early FL rounds (instead in the last round/s)
# Results for space_ga dataset in table 3
python -m hfedxgboost.main --multirun clients=space_ga_2_clients,space_ga_5_clients,space_ga_10_clients dataset=space_ga

# Results for abalone dataset in table 3
python -m hfedxgboost.main --multirun clients=abalone_2_clients,abalone_5_clients,abalone_10_clients dataset=abalone

# Results for cpusmall dataset in table 3
python -m hfedxgboost.main --multirun clients=cpusmall_2_clients,cpusmall_5_clients,cpusmall_10_clients dataset=cpusmall

# Results for YearPredictionMSD_2 dataset in table 3
python -m hfedxgboost.main --multirun clients=YearPredictionMSD_2_clients,YearPredictionMSD_5_clients,YearPredictionMSD_10_clients dataset=YearPredictionMSD
```

| Dataset | task type |no. of clients | server-side test MSE |
| :---: | :---: | :---: | :---: |
| space_ga | Regression | 2<br>5<br>10 | 0.024<br>0.033<br>0.034 |
| abalone | Regression | 2<br>5<br>10 | 5.5<br>6.87<br>7.5 | 
| cpusmall | Regression | 2<br>5<br>10 | 13<br>15.13<br>15.28 | 
| YearPredictionMSD | Regression | 2<br>5<br>10 | 119<br>118<br>118 | 


## Doing your own finetuning

There are 3 main things that you should consider:

1- You can use WandB to automate the fine-tuning process, modify the `sweep.yaml` file to control your experiments settings including your search methods, values to choose from, etc. Below we demonstrate how to run the `wandb` sweep.
If you're new to `wandb` you might want to read the following resources to [do hyperparameter tuning with W&B+PyTorch](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb), and [use W&B alongside Hydra](https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw).

```
# Remember to activate the poetry shell
poetry shell

# login to your wandb account
wandb login

# Inside the folder flower/baselines/hfedxgboost/hfedxgboost run the commands below

# Initiate WandB sweep
wandb sweep sweep.yaml

# that command -if ran with no error- will return a line that contains
# the command that you can use to run the sweep agent, it'll look something like that:

wandb agent <your user name on wandb>/flower-baselines_hfedxgboost_hfedxgboost/<the sweep name>

```

2- The config files named `<dataset name>_<no.of clients>_clients.yaml` are meant to keep the final hyperparameters values, so whenever you think you're done with fine-tuning some hyperparameters, add them to their config files so the one after you can use them.

3- To help with the fine-tuning of the hyperparameters process, there are 2 classes in the utils.py that write down the used hyperparameters in the experiments and the results for that experiment in 2 separate CSV files, some of the hyperparameters used in the experiments done during building this baseline can be found in results.csv and results_centralized.csv files.<br>
More important, those 2 classes focus on writing down only the hyperparameters that I thought was important so if you're interested in experimenting with other hyperparameters, don't forget to add them to the writers classes so you can track them more easily, especially if you intend to do some experiments away from WandB.


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

**2- Add config files for your dataset:**

**a- config files for the centralized baseline:**

- To run the centralized model on your dataset with the original hyper-parameters from the paper alongside all the other datasets added before just do the following step:
   - in the dictionary called `dataset_tasks` in the `utils.py` file add your dataset name as a key -the same name that you put in the `download_data` function  in the step before- and add its task type, this code performs for 2 tasks: `BINARY` which is binary classification or `REG` which is regression.
    
- To run the centralized model on your dataset you need to create a config file `<your dataset>.yaml` in the `xgboost_params_centralized` folder and another .yaml file in the `dataset` folder -you will find that one of course inside the `conf` folder :) - and you need to specify the hyper-parameters of your choice for the xgboost model

  - the .yaml file in the `dataset` folder should look something like this:
    ```
    defaults:
       - task: <config file name with your task type, "Regression" or "Binary_Classification">
    dataset_name: "<your dataset name, must be the same name used in the download_data function>"
    train_ratio: <percentage of the training data of the whole dataset>
    early_stop_patience_rounds: <no.of epochs that the early stopper class should wait before ending the training>
    ```
  - the .yaml file in the `xgboost_params_centralized` folder should contain the values for all the hyper-parameters of your choice for the xgboost model

You can skip this whole step and use the paper default hyper-parameters from the paper, they're all written in the "paper_<no. clients>_clients.yaml" files.<br>
**b- config files for the federated baseline:**

To run the federated baseline with your dataset using your customized hyper-parameters, you need first to create the .yaml file in the `dataset` folder that was mentioned before and you need to create config files that contain the no.of the clients and it should look something like this:
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
