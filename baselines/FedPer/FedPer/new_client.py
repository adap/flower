import flwr as fl
import torch

from torch.utils.data import DataLoader
from FedPer.new_models import CNNModelManager

from fedpfl.federated_learning.utils import get_client_cls, load_config

class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
    ):  # pylint: disable=too-many-arguments
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_id = 1

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        if self.net.split: 
            return [val.cpu().numpy() for _, val in self.net.body.state_dict().items()]
        else:
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

if __name__ == '__main__':

    config = load_config("../../data/config.json")

    client_cls = get_client_cls(algorithm=config.get("algorithm", "FedAvg"))

    client = client_cls(
        config=config,
        client_id=0,
        model_manager_class=CNNModelManager
    )

    fl.client.start_numpy_client("127.0.0.1:8080", client=client)