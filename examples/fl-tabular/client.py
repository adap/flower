from flwr.client import Client, ClientApp, NumPyClient
from flwr_datasets import FederatedDataset
from task import set_weights, get_weights, train, evaluate, IncomeClassifier, load_data

NUMBER_OF_CLIENTS = 5


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


def get_client_fn(dataset: FederatedDataset):
    def client_fn(cid: str) -> Client:
        train_loader, test_loader = load_data(partition_id=int(cid), fds=dataset)
        net = IncomeClassifier(14)
        return FlowerClient(net, train_loader, test_loader).to_client()

    return client_fn


fds = FederatedDataset(
    dataset="scikit-learn/adult-census-income",
    partitioners={"train": NUMBER_OF_CLIENTS},
)
app = ClientApp(client_fn=get_client_fn(fds))
