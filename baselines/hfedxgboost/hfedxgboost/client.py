"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
from typing import Any, Tuple, Union

import flwr as fl
import torch
import torch.nn as nn
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, MeanSquaredError
from tqdm import tqdm
from xgboost import XGBClassifier, XGBRegressor

from hfedxgboost.models import CNN, fit_XGBoost
from hfedxgboost.utils import single_tree_preds_from_each_client


class FL_Client(fl.client.Client):
    """Custom class contains the methods that the client need."""

    def __init__(
        self,
        cfg,
        trainloader: DataLoader,
        valloader: DataLoader,
        client_num: int,
        cid: str,
        log_progress: bool = False,
    ):
        self.task_type = cfg.dataset.task.task_type
        self.cid = cid
        self.config = cfg
        for dataset in trainloader:
            data, label = dataset[0], dataset[1]
        self.tree = fit_XGBoost(cfg, self.task_type, data, label, 100)

        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.valloader: Any
        self.n_estimators_client = (
            cfg.n_estimators_client
        )  # 100#cfg.dataset.n_estimators_client
        self.client_num = client_num
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress

        # instantiate model
        self.net = CNN(cfg)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.task_type == "BINARY":
            self.metric_fn = Accuracy(task="binary")
            self.metric_name = "accuracy"
            self.criterion = nn.BCELoss()
        elif self.task_type == "REG":
            self.metric_fn = MeanSquaredError()
            self.metric_name = "mse"
            self.criterion = nn.MSELoss()
        else:
            raise Exception("choose a valid task type, BINARY or REG")
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=cfg.clients.CNN.lr, betas=(0.9, 0.999)
        )

    def train_one_loop(self, data):
        """Trains the neural network model for one loop iteration.

        Parameters
        ----------
            data (tuple): A tuple containing the inputs and
            labels for the training data, where the input represent the predictions
            of the trees from the tree ensemples of the clients.

        Returns
        -------
            loss (float): The value of the loss function after the iteration.
            metric_val * n_samples (float): The value of the chosen evaluation metric
            (accuracy or MSE) after the iteration.
            n_samples (int): The number of samples used for training in the iteration.
        """
        tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
        self.optimizer.zero_grad()

        outputs = self.net(tree_outputs)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        # Collected training loss and accuracy statistics
        n_samples = labels.size(0)
        metric_val = self.metric_fn(outputs, labels.type(torch.int))

        return loss.item(), metric_val * n_samples, n_samples

    def train(
        self,
        net: CNN,
        trainloader: DataLoader,
        num_iterations: int,
        log_progress: bool = True,
    ) -> Tuple[float, float, int]:
        """Train CNN model on a given dataset(trainloader) for(num_iterations).

        Parameters
        ----------
            net (CNN): The convolutional neural network to be trained.
            trainloader (DataLoader): The data loader object containing
             the training dataset.
            num_iterations (int): The number of iterations or batches to
             be processed by the network.
            log_progress (bool, optional): Set to True if you want to log the
             training progress using tqdm progress bar. Default is True.

        Returns
        -------
            Tuple[float, float, int]: A tuple containing the average loss per sample,
            the average evaluation result per sample, and the total number of training
            samples processed.

        Note:

            - The training is formulated in terms of the number of updates or iterations
            processed by the network.
            - If log_progress is set to True, it displays the training progress with a
            progress bar and prints the average loss and evaluation result per sample.
        """
        net.train()
        total_loss, total_result, total_n_samples = 0.0, 0.0, 0
        progress_bar = (
            tqdm(iter(trainloader), total=num_iterations, desc="TRAIN")
            if log_progress
            else iter(trainloader)
        )

        # Unusually, this training is formulated in terms of number of
        # updates/iterations/batches processed
        # by the network. This will be helpful later on, when partitioning the
        # data across clients: resulting
        # in differences between dataset sizes and hence inconsistent numbers of updates
        # per 'epoch'.
        for _i, data in zip(range(num_iterations), progress_bar):
            loss, metric_val, n_samples = self.train_one_loop(data)
            total_loss += loss
            total_result += metric_val
            total_n_samples += n_samples
            if log_progress:
                progress_bar.set_postfix(
                    {
                        "train_loss": total_loss / n_samples,
                        "train_" + self.metric_name: total_result / n_samples,
                    }
                )
        if log_progress:
            print("\n")

        return (
            total_loss / total_n_samples,
            total_result / total_n_samples,
            total_n_samples,
        )

    def test(self, net: CNN, testloader: DataLoader) -> Tuple[float, float, int]:
        """Evaluates the network on test data.

        Parameters
        ----------
         net: The CNN model to be tested.
         testloader: The data loader containing the test data.

         Return: A tuple containing the average loss,
         average metric result,
         and the total number of samples tested.
        """
        total_loss, total_result, n_samples = 0.0, 0.0, 0
        net.eval()
        with torch.no_grad():
            progress_bar = (
                tqdm(testloader, desc="TEST") if self.log_progress else testloader
            )
            for data in progress_bar:
                tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(tree_outputs)
                total_loss += self.criterion(outputs, labels).item()
                n_samples += labels.size(0)
                metric_val = self.metric_fn(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += metric_val * labels.size(0)
        if self.log_progress:
            print("\n")

        return total_loss / n_samples, total_result / n_samples, n_samples

    def get_parameters(
        self, ins: GetParametersIns
    ) -> Tuple[
        GetParametersRes,
        Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]],
    ]:
        """Get CNN net weights and the tree.

        Parameters
        ----------
        - self (object): The instance of the class that the function belongs to.
        - ins (GetParametersIns): An input parameter object.

        Returns
        -------
        Tuple[GetParametersRes,
        Union[Tuple[XGBClassifier, int],Tuple[XGBRegressor, int]]]:
            A tuple containing the parameters of the net and the tree.
            - GetParametersRes:
                - status : An object with the status code.
                - parameters : An ndarray containing the model's weights.
            - Union[Tuple[XGBClassifier, int], Tuple[XGBRegressor, int]]:
                A tuple containing either an XGBClassifier or XGBRegressor
                object along with client's id.
        """
        return GetParametersRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(self.net.get_weights()),
        ), (self.tree, int(self.cid))

    def fit(self, fit_params: FitIns) -> FitRes:
        """Trains a model using the given fit parameters.

        Parameters
        ----------
            fit_params: FitIns - The fit parameters that contain the configuration
            and parameters needed for training.

        Returns
        -------
            FitRes - An object that contains the status, trained parameters,
            number of examples processed, and metrics.
        """
        num_iterations = fit_params.config["num_iterations"]
        batch_size = fit_params.config["batch_size"]

        # set parmeters
        self.net.set_weights(parameters_to_ndarrays(fit_params.parameters[0]))
        aggregated_trees = fit_params.parameters[1]

        if isinstance(aggregated_trees, list):
            print("Client " + self.cid + ": recieved", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        trainloader: Any = single_tree_preds_from_each_client(
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
        # num_iterations = None special behaviour: train(...)
        # runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(trainloader)
        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_result, num_examples = self.train(
            self.net,
            trainloader,
            num_iterations=num_iterations,
            log_progress=self.log_progress,
        )
        print(
            f"Client {self.cid}: training round complete, {num_examples}",
            "examples processed",
        )

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(fit_params.config),
            num_examples=num_examples,
            metrics={"loss": train_loss, self.metric_name: train_result},
        )

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        """Evaluate CNN model using the given evaluation parameters.

        Parameters
        ----------
          eval_params: An instance of EvaluateIns class that contains the parameters
          for evaluation.
        Return:
          An EvaluateRes object that contains the evaluation results.
        """
        # set the weights of the CNN net
        self.net.set_weights(parameters_to_ndarrays(eval_params.parameters))

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = self.test(
            self.net,
            self.valloader,
        )

        # Return evaluation information
        print(
            f"Client {self.cid}: evaluation on {num_examples} examples:",
            f"loss={loss:.4f}, {self.metric_name}={result:.4f}",
        )
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss,
            num_examples=num_examples,
            metrics={self.metric_name: result},
        )
