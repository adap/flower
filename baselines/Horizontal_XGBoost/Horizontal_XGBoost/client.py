"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    Code,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)

from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchmetrics import Accuracy, MeanSquaredError
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, Dataset, random_split
import flwr as fl
from flwr.common.typing import Parameters
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union
from flwr.common import NDArray, NDArrays
from torch.utils.data import TensorDataset
from .utils import get_dataloader,single_tree_preds_from_each_client
from .models import fit_XGBoost,CNN

class FL_Client(fl.client.Client):
    def __init__(
        self,
        cfg,
        trainloader: DataLoader,
        valloader: DataLoader,
        client_num: int,
        cid: str,
        log_progress: bool = False,
    ):
        """
        Creates a client for training `network.Net` on tabular dataset.
        """
        self.task_type = cfg.dataset.task_type
        self.cid = cid
        self.config=cfg
        for dataset in trainloader:
            data, label = dataset[0], dataset[1]
        self.tree = fit_XGBoost(cfg,self.task_type,data, label,100)

        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.trainloader = None
        self.valloader = None
        self.n_estimators_client = cfg.dataset.n_estimators_client#100#cfg.dataset.n_estimators_client
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress

        # instantiate model
        self.net = CNN(cfg)

        # determine device
        self.device = "cpu"#torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.task_type == "BINARY":
            self.metric_fn=Accuracy(task="binary")
            self.metric_name="accuracy"
            self.criterion = nn.BCELoss()
        elif self.task_type == "REG":
            self.metric_fn=MeanSquaredError()
            self.metric_name="mse"
            self.criterion = nn.MSELoss()
        else:
            raise Exception(
                    "choose a valid task type, BINARY or REG"
                )
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=cfg.dataset.CNN.lr, betas=(0.9, 0.999))

    def train_one_loop(self,data):
                tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
                self.optimizer.zero_grad()

                outputs = self.net(tree_outputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Collected training loss and accuracy statistics
                #total_loss += loss.item()
                n_samples = labels.size(0)
                metric_val=self.metric_fn(outputs, labels.type(torch.int))
                #total_result += metric_val * n_samples


                return loss.item(), metric_val * n_samples, n_samples
    def train(
            self,
            net: CNN,
            trainloader: DataLoader,
            num_iterations: int,
            log_progress: bool = True,
        ) -> Tuple[float, float, int]:
            # Train the network
            net.train()
            total_loss, total_result, total_n_samples = 0.0, 0.0, 0
            pbar = (
                tqdm(iter(trainloader), total=num_iterations, desc=f"TRAIN")
                if log_progress
                else iter(trainloader)
            )

            # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
            # by the network. This will be helpful later on, when partitioning the data across clients: resulting
            # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
            for i, data in zip(range(num_iterations), pbar):
                loss,metric_val,n_samples=self.train_one_loop(data)
                total_loss+=loss
                total_result+=metric_val
                total_n_samples+=n_samples
                if log_progress:
                        pbar.set_postfix(
                            {
                                "train_loss": total_loss / n_samples,
                                "train_"+self.metric_name: total_result / n_samples,
                            }
                        )
            if log_progress:
                print("\n")

            return total_loss / total_n_samples, total_result / total_n_samples, total_n_samples
    def test(
            self,
            net: CNN,
            testloader: DataLoader
        ) -> Tuple[float, float, int]:
            """Evaluates the network on test data."""
            total_loss, total_result, n_samples = 0.0, 0.0, 0
            net.eval()
            with torch.no_grad():
                pbar = tqdm(testloader, desc="TEST") if self.log_progress else testloader
                for data in pbar:
                    tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
                    outputs = net(tree_outputs)
                    total_loss += self.criterion(outputs, labels).item()
                    n_samples += labels.size(0)
                    metric_val=self.metric_fn(outputs.cpu(), labels.type(torch.int).cpu())
                    total_result += metric_val * labels.size(0)
            if self.log_progress:
                print("\n")

            return total_loss / n_samples, total_result / n_samples, n_samples
    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(
        self, ins: GetParametersIns
    ) -> Tuple[
        GetParametersRes, Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]
    ]:
        """
        Get the weights of the trained CNN model and the trained xgboost tree of the client
        """
        return [
            GetParametersRes(
                status=Status(Code.OK, ""),
                parameters=ndarrays_to_parameters(self.net.get_weights())),
            (self.tree, int(self.cid)),
        ]

    def set_parameters(
        self,
        parameters: Tuple[
            Parameters,
            Union[
                Tuple[XGBClassifier, int],
                Tuple[XGBRegressor, int],
                List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
            ],
        ],
    ) -> Union[
        Tuple[XGBClassifier, int],
        Tuple[XGBRegressor, int],
        List[Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]],
    ]:
        self.net.set_weights(parameters_to_ndarrays(parameters[0]))
        return parameters[1]

    def fit(self, fit_params: FitIns) -> FitRes:
        # Process incoming request to train
        num_iterations = fit_params.config["num_iterations"]
        batch_size = fit_params.config["batch_size"]
        aggregated_trees = self.set_parameters(fit_params.parameters)

        if type(aggregated_trees) is list:
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        self.trainloader = single_tree_preds_from_each_client(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.n_estimators_client,
            self.client_num,
        )
        self.valloader = single_tree_preds_from_each_client(
            self.valloader_original,
            batch_size,
            aggregated_trees,
            self.n_estimators_client,
            self.client_num,
        )
        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(self.trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_result, num_examples = self.train(
            self.net,
            self.trainloader,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
        )
        print(
            f"Client {self.cid}: training round complete, {num_examples} examples processed"
        )

        # Return training information: model, number of examples processed and metrics
        return FitRes(
                status=Status(Code.OK, ""),
                parameters=self.get_parameters(fit_params.config),
                num_examples=num_examples,
                metrics={"loss": train_loss, self.metric_name: train_result},
            )


    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        self.set_parameters(eval_params.parameters)

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = self.test(
            self.net,
            self.valloader,
        )

        # Return evaluation information
        print(
                f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, {self.metric_name}={result:.4f}"
            )
        return EvaluateRes(
                status=Status(Code.OK, ""),
                loss=loss,
                num_examples=num_examples,
                metrics={self.metric_name: result},
            )