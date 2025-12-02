import torch
from flwr.client import NumPyClient
from flwr.clientapp import ClientApp
from flwr.common import Context
from sklearn.preprocessing import StandardScaler

from vertical_fl.task import ClientModel, load_data


import torch
from flwr.app import Array, ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp

# Flower ClientApp
app = ClientApp()


@app.query("generate_embeddings")
def f(msg: Message, context: Context):
    """Generate embeddings."""

    # Read from config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    #! feature_splits: str = context.run_config["feature-splits"]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    #! in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]
    data, v_split_id = load_data(partition_id, num_partitions=num_partitions)

    assert v_split_id == partition_id

    data = torch.tensor(StandardScaler().fit_transform(data)).float()
    model = ClientModel(input_size=data.shape[1], out_feat_dim=out_feature_dim_clientapp)

    # Do forward pass
    embedding = model(data)

    # Construct and return reply Message
    model_record = ArrayRecord({"embedding": Array(embedding.detach().numpy())})
    content = RecordDict(
        {
            "arrays": model_record,
            "config": ConfigRecord({"pos": partition_id}),
        }
    )
    return Message(content=content, reply_to=msg)

@app.train("apply_gradients")
def f(msg: Message, context: Context):
    """Apply gradients to local model."""

    # Read from config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    #! feature_splits: str = context.run_config["feature-splits"]
    #! in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]
    lr = context.run_config["learning-rate"]
    data, v_split_id = load_data(partition_id, num_partitions=num_partitions)

    assert v_split_id == partition_id

    data = torch.tensor(StandardScaler().fit_transform(data)).float()
    model = ClientModel(input_size=data.shape[1], out_feat_dim=out_feature_dim_clientapp)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Do forward pass
    embedding = model(data)

    # Get gradients from message and apply them
    embedding.backward(
        torch.from_numpy(msg.content["gradients"]["local-gradients"].numpy())
    )
    optimizer.step()

    # Save updated model in state for next round
    context.state["model"] = ArrayRecord(model.state_dict())

    # Construct and return reply Message
    return Message(content=RecordDict(), reply_to=msg)

# class FlowerClient(NumPyClient):
#     def __init__(self, v_split_id, data, lr):
#         self.v_split_id = v_split_id
#         self.data = torch.tensor(StandardScaler().fit_transform(data)).float()
#         self.model = ClientModel(input_size=self.data.shape[1])
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

#     def get_parameters(self, config):
#         pass

#     def fit(self, parameters, config):
#         embedding = self.model(self.data)
#         return [embedding.detach().numpy()], 1, {}

#     def evaluate(self, parameters, config):
#         self.model.zero_grad()
#         embedding = self.model(self.data)
#         embedding.backward(torch.from_numpy(parameters[int(self.v_split_id)]))
#         self.optimizer.step()
#         return 0.0, 1, {}


# def client_fn(context: Context):
#     partition_id = context.node_config["partition-id"]
#     num_partitions = context.node_config["num-partitions"]
#     partition, v_split_id = load_data(partition_id, num_partitions=num_partitions)
#     lr = context.run_config["learning-rate"]
#     return FlowerClient(v_split_id, partition, lr).to_client()


# app = ClientApp(
#     client_fn=client_fn,
# )
