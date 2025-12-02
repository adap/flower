
from logging import INFO
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import Strategy
from vertical_fl.task import process_dataset, ServerModel


# def server_fn(context: Context) -> ServerAppComponents:
#     """Construct components that set the ServerApp behaviour."""

#     # Get dataset
#     processed_df, _ = process_dataset()

#     # Define the strategy
#     strategy = Strategy(processed_df["Survived"].values)

#     # Construct ServerConfig
#     num_rounds = context.run_config["num-server-rounds"]
#     config = ServerConfig(num_rounds=num_rounds)

#     return ServerAppComponents(strategy=strategy, config=config)


# # Start Flower server
# app = ServerApp(server_fn=server_fn)

import torch
from flwr.app import Array, ArrayRecord, Context, Message, RecordDict
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    num_rounds: int = context.run_config["num-server-rounds"]
    feature_splits: str = context.run_config["feature-splits"]
    in_feature_dim_clientapp = [int(dim) for dim in feature_splits.split(",")]
    out_feature_dim_clientapp: int = context.run_config["out-feature-dim-clientapp"]


    # Get dataset
    processed_df, _ = process_dataset()
    labels = processed_df["Survived"].values
    labels = torch.tensor(labels).float().unsqueeze(1)


    # Serverapp model
    head = None
    optimizer = None
    criterion = torch.nn.BCELoss()

    for i in range(num_rounds):
        log(INFO, "")
        log(INFO, f"--- ServerApp Round {i + 1} ---")

        node_ids = list(grid.get_node_ids())

        if len(node_ids) != len(in_feature_dim_clientapp):
            raise ValueError(
                "Number of feature splits must be equal to number of nodes."
                f" Got {len(in_feature_dim_clientapp)} splits and {len(node_ids)} nodes."
            )
        
        if head is None:
            # The server model's input size is determined by the number of clients
            # and the output feature dimension of each embedding produced by the clients
            head = ServerModel(input_size=out_feature_dim_clientapp * len(node_ids))
            optimizer = torch.optim.SGD(head.parameters(), lr=0.001)

        optimizer.zero_grad()

        # Get embeddings from all clients
        embeddings, node_pos_mapping = get_remote_embeddings(
            grid=grid,
            node_ids=node_ids,
            num_nodes=len(processed_df),
            embedding_dim=out_feature_dim_clientapp,
        )


        # Complete forward pass and compute loss
        output = head(embeddings)
        loss = criterion(output, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
    
        with torch.no_grad():
            correct = 0
            output = head(embeddings)
            predicted = (output > 0.5).float()

            correct += (predicted == labels).sum().item()

            accuracy = correct / len(labels) * 100

        print(f"Round {i+1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")


        # # Compute gradients w.r.t. embeddings
        # embeddings_grad = embeddings.grad.split(
        #     [out_feature_dim_clientapp] * len(node_ids), dim=1
        # )
        # messages = []
        # for node_id, pos in node_pos_mapping.items():
        #     arrc = ArrayRecord({"local-gradients": Array(embeddings_grad[pos].numpy())})
        #     message = Message(
        #         content=RecordDict({"gradients": arrc}),
        #         message_type="train.apply_gradients",  # target `query` method in ClientApp
        #         dst_node_id=node_id,
        #     )
        #     messages.append(message)

        # # Send messages and wait for all results
        # log(INFO, "Sending gradients to %s nodes...", len(messages))
        # replies = grid.send_and_receive(messages)
        # log(INFO, "\tReceived %s/%s results", len(replies), len(messages))

        # # # Send gradients to clients

        # log(INFO, f"--- End of Round {i + 1} ---")




def get_remote_embeddings(
    grid: Grid,
    node_ids: list[str],
    num_nodes: int,
    embedding_dim: int,
) -> tuple[torch.Tensor, dict[int, int]]:
    """Get embeddings from sampled remote nodes."""

    # Create messages
    messages = []
    for node_id in node_ids:  # one message for each node
        message = Message(
            content=RecordDict(),
            message_type="query.generate_embeddings",
            dst_node_id=node_id,
        )
        messages.append(message)

    # Send messages and wait for all results
    log(INFO, "Requesting embeddings from %s nodes...", len(messages))
    replies = grid.send_and_receive(messages)
    log(INFO, "\tReceived %s/%s results", len(replies), len(messages))

    embeddings = torch.zeros((num_nodes, embedding_dim * len(node_ids)))

    # Convert all embeddings back to pytorch tensors
    # and place them in the corresponding feature segment
    node_pos_mapping: dict[int, int] = {}
    for reply in replies:
        if reply.has_content():
            arr = reply.content["arrays"]["embedding"]
            embd = torch.from_numpy(arr.numpy())
            pos = reply.content["config"]["pos"]
            node_pos_mapping[reply.metadata.src_node_id] = pos
            embeddings[:, pos * embedding_dim : (pos + 1) * embedding_dim] = embd

    return embeddings.requires_grad_(), node_pos_mapping
