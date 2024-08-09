from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn.preprocessing import StandardScaler
import torch

from vertical_fl.task import ClientModel, get_partitions_and_label


class FlowerClient(NumPyClient):
    def __init__(self, cid, data, lr):
        self.cid = cid
        self.train = torch.tensor(StandardScaler().fit_transform(data)).float()
        self.model = ClientModel(input_size=self.train.shape[1])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
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


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    lr = context.run_config["learning-rate"]
    return FlowerClient(partition_id, partitions[partition_id], lr).to_client()


app = ClientApp(
    client_fn=client_fn,
)
