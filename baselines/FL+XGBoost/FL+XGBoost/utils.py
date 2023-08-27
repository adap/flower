"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
from sklearn.metrics import mean_squared_error, accuracy_score
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from dataset import load_single_dataset

def evaluate(task_type,y,preds):
    if task_type.upper() == "BINARY":
        result = accuracy_score(y, preds)
    elif task_type.upper() == "REG":
        result = mean_squared_error(y, preds)
    return result

def run_single_exp(config,dataset_name,task_type):
    X_train,y_train,X_test,y_test=load_single_dataset(task_type,dataset_name,train_ratio=config.dataset.train_ratio)
    if task_type.upper() == "REG":
        tree = instantiate(config.XGBoost.regressor)
    elif task_type.upper() == "BINARY":
        tree = instantiate(config.XGBoost.classifier)
    tree.fit(X_train, y_train)
    preds_train = tree.predict(X_train)
    result_train=evaluate(task_type,y_train,preds_train)
    preds_test = tree.predict(X_test)
    result_test=evaluate(task_type,y_test,preds_test)
    return result_train,result_test
def run_exp(config,dataset_name="all",task_type=None):
    dataset_tasks={
        "a9a":"BINARY",
        "cod-rna":"BINARY",
        "ijcnn1":"BINARY",
        "abalone":"REG",
        "cpusmall":"REG",
        "space_ga":"REG"
    }
    if dataset_name=="all":
        for dataset in dataset_tasks:
            result_train,result_test=run_single_exp(config,dataset,dataset_tasks[dataset])
            print("Results for",dataset,", Task:",dataset_tasks[dataset],", Train:",result_train,", Test:",result_test)
    else:
        if task_type:
            result_train,result_test=run_single_exp(config,dataset,task_type)
            print("Results for",dataset,", Task:",task_type,", Train:",result_train,", Test:",result_test)
        else:
            if dataset_name in dataset_tasks.keys:
                result_train,result_test=run_single_exp(config,dataset,dataset_tasks[dataset])
                print("Results for",dataset,", Task:",dataset_tasks[dataset],", Train:",result_train,", Test:",result_test)
            else:
                raise Exception(
                    "task_type should be assigned to be BINARY for binary classification tasks" 
                    "or REG for regression tasks"
                    "or the dataset should be one of the follwing"
                    "a9a, cod-rna, ijcnn1, space_ga, abalone, cpusmall"
                    )