from flwr.client import Client, ClientApp, NumPyClient
from task import set_weights, get_weights, train, evaluate, IncomeClassifier, load_data


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(cid: str) -> Client:
    train_loader, test_loader = load_data(partition_id=int(cid))
    net = IncomeClassifier()
    return FlowerClient(net, train_loader, test_loader).to_client()


app = ClientApp(client_fn=client_fn)
