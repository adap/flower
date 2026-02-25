"""Flower ClientApp for federated NanoGPT."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from nanogpt_shakespeare.task import build_model, load_data, load_local_data
from nanogpt_shakespeare.task import test as test_fn
from nanogpt_shakespeare.task import train as train_fn

app = ClientApp()


def _load_trainval(context: Context):
    """Load train/val data for either Simulation or Deployment."""
    cfg = context.run_config
    batch_size = int(cfg["batch-size"])
    block_size = int(cfg["block-size"])

    if (
        "partition-id" in context.node_config
        and "num-partitions" in context.node_config
    ):
        # Simulation Engine: partition data on the fly
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        return load_data(partition_id, num_partitions, batch_size, block_size)
    else:
        # Deployment Engine: load pre-staged data from disk
        data_path = context.node_config["data-path"]
        return load_local_data(data_path, batch_size, block_size)


@app.train()
def train(msg: Message, context: Context):
    """Train NanoGPT on this client's data."""
    cfg = context.run_config
    model = build_model(cfg)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, _ = _load_trainval(context)

    train_loss = train_fn(
        model,
        trainloader,
        epochs=int(cfg["local-epochs"]),
        lr=msg.content["config"]["lr"],
        device=str(device),
        max_steps=int(cfg["max-steps"]),
    )

    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    content = RecordDict({"arrays": model_record, "metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate NanoGPT on this client's validation data."""
    cfg = context.run_config
    model = build_model(cfg)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _, valloader = _load_trainval(context)

    eval_loss, eval_ppl = test_fn(model, valloader, str(device))

    metrics = {
        "eval_loss": eval_loss,
        "perplexity": eval_ppl,
        "num-examples": len(valloader.dataset),
    }
    content = RecordDict({"metrics": MetricRecord(metrics)})
    return Message(content=content, reply_to=msg)
