"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from typing import Any, Tuple

import flwr as fl
import torch
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hfedxgboost.models import CNN, fit_xgboost
from hfedxgboost.utils import single_tree_preds_from_each_client


class FlClient(fl.client.Client):
    """Custom class contains the methods that the client need."""

    def __init__(
        self,
        cfg: DictConfig,
        trainloader: DataLoader,
        valloader: DataLoader,
        cid: str,
    ):
        self.cid = cid
        self.config = cfg

        self.trainloader_original = trainloader
        self.valloader_original = valloader
        self.valloader: Any

        # instantiate model
        self.net = CNN(cfg)

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train_one_loop(self, data, optimizer, metric_fn, criterion):
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
        optimizer.zero_grad()

        outputs = self.net(tree_outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        n_samples = labels.size(0)
        metric_val = metric_fn(outputs, labels.type(torch.int))

        return loss.item(), metric_val * n_samples, n_samples

    def train(
        self,
        net: CNN,
        trainloader: DataLoader,
        num_iterations: int,
    ) -> Tuple[float, float, int]:
        """Train CNN model on a given dataset(trainloader) for(num_iterations).

        Parameters
        ----------
            net (CNN): The convolutional neural network to be trained.
            trainloader (DataLoader): The data loader object containing
             the training dataset.
            num_iterations (int): The number of iterations or batches to
             be processed by the network.

        Returns
        -------
            Tuple[float, float, int]: A tuple containing the average loss per sample,
            the average evaluation result per sample, and the total number of training
            samples processed.

        Note:

            - The training is formulated in terms of the number of updates or iterations
            processed by the network.
        """
        net.train()
        total_loss, total_result, total_n_samples = 0.0, 0.0, 0

        # Unusually, this training is formulated in terms of number of
        # updates/iterations/batches processed
        # by the network. This will be helpful later on, when partitioning the
        # data across clients: resulting
        # in differences between dataset sizes and hence inconsistent numbers of updates
        # per 'epoch'.
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.config.clients.CNN.lr, betas=(0.5, 0.999)
        )
        metric_fn = instantiate(self.config.dataset.task.metric.fn)
        criterion = instantiate(self.config.dataset.task.criterion)
        for _i, data in zip(range(num_iterations), trainloader):
            loss, metric_val, n_samples = self.train_one_loop(
                data, optimizer, metric_fn, criterion
            )
            total_loss += loss
            total_result += metric_val
            total_n_samples += n_samples

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
        metric_fn = instantiate(self.config.dataset.task.metric.fn)
        criterion = instantiate(self.config.dataset.task.criterion)
        with torch.no_grad():
            for data in testloader:
                tree_outputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = net(tree_outputs)
                total_loss += criterion(outputs, labels).item()
                n_samples += labels.size(0)
                metric_val = metric_fn(outputs.cpu(), labels.type(torch.int).cpu())
                total_result += metric_val * labels.size(0)

        return total_loss / n_samples, total_result / n_samples, n_samples

    def get_parameters(self, ins):
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
        for dataset in self.trainloader_original:
            data, label = dataset[0], dataset[1]

        tree = fit_xgboost(
            self.config, self.config.dataset.task.task_type, data, label, 100
        )
        return GetParametersRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(self.net.get_weights()),
        ), (tree, int(self.cid))

    def fit(self, ins: FitIns) -> FitRes:
        """Trains a model using the given fit parameters.

        Parameters
        ----------
            ins: FitIns - The fit parameters that contain the configuration
            and parameters needed for training.

        Returns
        -------
            FitRes - An object that contains the status, trained parameters,
            number of examples processed, and metrics.
        """
        num_iterations = ins.config["num_iterations"]
        batch_size = ins.config["batch_size"]

        # set parmeters
        self.net.set_weights(parameters_to_ndarrays(ins.parameters[0]))  # type: ignore # noqa: E501 # pylint: disable=line-too-long
        aggregated_trees = ins.parameters[1]  # type: ignore # noqa: E501 # pylint: disable=line-too-long

        if isinstance(aggregated_trees, list):
            print("Client " + self.cid + ": received", len(aggregated_trees), "trees")
        else:
            print("Client " + self.cid + ": only had its own tree")
        trainloader: Any = single_tree_preds_from_each_client(
            self.trainloader_original,
            batch_size,
            aggregated_trees,
            self.config.n_estimators_client,
            self.config.clients.client_num,
        )
        self.valloader = single_tree_preds_from_each_client(
            self.valloader_original,
            batch_size,
            aggregated_trees,
            self.config.n_estimators_client,
            self.config.clients.client_num,
        )

        # runs for a single epoch, however many updates it may be
        num_iterations = int(num_iterations) or len(trainloader)
        # Train the model
        print(
            "Client", self.cid, ": training for", num_iterations, "iterations/updates"
        )
        self.net.to(self.device)
        train_loss, train_result, num_examples = self.train(
            self.net,
            trainloader,
            num_iterations=num_iterations,
        )
        print(
            f"Client {self.cid}: training round complete, {num_examples}",
            "examples processed",
        )

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(ins.config),
            num_examples=num_examples,
            metrics={
                "loss": train_loss,
                self.config.dataset.task.metric.name: train_result,
            },
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate CNN model using the given evaluation parameters.

        Parameters
        ----------
          ins: An instance of EvaluateIns class that contains the parameters
          for evaluation.
        Return:
          An EvaluateRes object that contains the evaluation results.
        """
        # set the weights of the CNN net
        self.net.set_weights(parameters_to_ndarrays(ins.parameters))

        # Evaluate the model
        self.net.to(self.device)
        loss, result, num_examples = self.test(
            self.net,
            self.valloader,
        )

        # Return evaluation information
        print(
            f"Client {self.cid}: evaluation on {num_examples} examples:",
            f"loss={loss:.4f}",
            self.config.dataset.task.metric.name,
            f"={result:.4f}",
        )
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss,
            num_examples=num_examples,
            metrics={self.config.dataset.task.metric.name: result},
        )
