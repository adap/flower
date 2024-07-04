from flwr.client import ClientApp, NumPyClient
import torch
from sklearn.preprocessing import StandardScaler

from vertical_fl.task import ClientModel, get_partitions_and_label


class FlowerClient(NumPyClient):
    def __init__(self, cid, data):
        self.cid = cid
        self.train = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.train.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.embedding = self.model(self.train)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        self.embedding = self.model(self.train)
        return [self.embedding.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.model.zero_grad()
        self.embedding.backward(torch.from_numpy(parameters[int(self.cid)]))
        self.optimizer.step()
        return 0.0, 1, {}


partitions, label = get_partitions_and_label()


def client_fn(node_id, partition_id):
    return FlowerClient(partition_id, partitions[partition_id]).to_client()


app = ClientApp(
    client_fn=client_fn,
)
