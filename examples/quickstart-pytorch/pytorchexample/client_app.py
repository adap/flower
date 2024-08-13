"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, RecordSet, ConfigsRecord

from pytorchexample.task import Net, get_weights, load_data, set_weights, test, train


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(
        self, trainloader, valloader, local_epochs, learning_rate, state: RecordSet
    ):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # pin state and init metricrecords entry if it doesn't exists
        # ! NumPyClient objects have a .state attribute (of type common.Context) that's being injected
        # ! behind the scenes. This is a legacy feature that's going to be deprecated in the next release
        # ! For now, please pass the context (or context.state) to the client constructor.
        self.local_state = state
        self.record_name = "fit_results"
        if self.record_name not in self.local_state.configs_records: # init if doesn't exist
            # There are diffent types of records, `ConfigsRecord` is the most versatile.
            self.local_state.configs_records[self.record_name] = ConfigsRecord()

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        # Append to state the results train() returned
        record_results = self.local_state.configs_records[self.record_name]
        for k,v in results.items():
            if k in record_results:
                record_results[k].append(v)
            else:
                record_results[k] = [v]
        # Will print all results in all ConfigRecords in the state
        print(self.local_state.configs_records)

        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(partition_id, num_partitions, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(
        trainloader,
        valloader,
        local_epochs,
        learning_rate,
        state=context.state,  #! Pass the state (and object of type common.RecordSet)
    ).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
