"""fedvaeexample: A Flower / PyTorch app for Federated Variational Autoencoder."""

from fedvaeexample.task import Net, load_data, set_weights, test, train

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context


class CifarClient(NumPyClient):
    def __init__(self, trainloader, testloader):
        self.net = Net()
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testloader = load_data(partition_id, num_partitions)

    return CifarClient(trainloader, testloader).to_client()


app = ClientApp(client_fn=client_fn)
