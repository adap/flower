"""bouquetfl: A Flower / PyTorch app."""

import json
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common import Context

from bouquetfl.core.emulation_engine import run_emulation

logger = logging.getLogger(__name__)
app = ClientApp()

# Accumulates per-client timing across rounds (persists for the lifetime of
# the Ray actor, i.e. across all rounds for the same client).
_timing_history: dict[int, list[dict]] = {}


@app.train()
def train(msg: Message, context: Context):
    """Run one local training round inside a hardware-restricted emulation."""

    client_id  = context.node_config["partition-id"]
    run_config = context.run_config

    # Save incoming global params to /tmp for the worker
    state_dict        = msg.content["arrays"].to_torch_state_dict()
    input_params_path = f"/tmp/global_params_{client_id}.pt"
    torch.save(state_dict, input_params_path)

    # Hardware profile for this client and local machine info, sent by server
    hardware_config  = json.loads(msg.content["config"]["hardware_config"])
    hardware_profile = hardware_config[f"client_{client_id}"]
    local_hw         = json.loads(msg.content["config"]["local_hw"])

    config = {
        "task":              f"task/{run_config['experiment']}.py",
        "client_id":         client_id,
        "num-clients":       run_config["num-clients"],
        "batch-size":        run_config["batch-size"],
        "local-epochs":      run_config["local-epochs"],
        "learning-rate":     run_config["learning-rate"],
        "server-round":      msg.content["config"]["server-round"],
        "num-server-rounds": run_config["num-server-rounds"],
    }

    timing, state_dict_updated = run_emulation(
        config=config,
        hardware_profile=hardware_profile,
        local_hw=local_hw,
        input_params_path=input_params_path,
        output_params=True,
    )

    os.remove(input_params_path)

    # On OOM the worker produces no parameters. We deliberately send back
    # the unmodified global model (received at the start of this round) so
    # FedAvg can still aggregate across all clients. num_examples is reported
    # as 0, it does not affect aggregation. Timing is still reported with oom=1.
    if state_dict_updated is None:
        state_dict_updated = state_dict

    num_examples = timing["num_examples"] if timing else 0
    server_round = msg.content["config"]["server-round"]

    # Accumulate timing and print per-client running average
    entry = _timing_history.setdefault(client_id, [])
    if timing:
        entry.append(timing)
    n         = len(entry)
    avg_load  = sum(t["data_load_time"] for t in entry) / n if n else 0.0
    avg_train = sum(t["train_time"]     for t in entry if not t.get("oom")) / max(1, sum(1 for t in entry if not t.get("oom")))
    oom_count = sum(1 for t in entry if t.get("oom"))
    print(
        f"[client {client_id}] round {server_round} — "
        f"load: {timing['data_load_time']:.2f}s  train: {timing['train_time']:.2f}s  "
        f"{'OOM  ' if timing.get('oom') else 'OK   '}"
        f"| avg over {n} round(s): load={avg_load:.2f}s  train={avg_train:.2f}s  oom={oom_count}/{n}"
        if timing else f"[client {client_id}] round {server_round} — no timing data"
    )

    metrics = {
        "data_load_time": timing["data_load_time"] if timing else -1.0,
        "train_time":     timing["train_time"]     if timing else -1.0,
        "oom":            int(timing.get("oom", False)) if timing else 1,
        "num-examples":   float(num_examples),
    }

    content = RecordDict({
        "arrays":  ArrayRecord(torch_state_dict=state_dict_updated),
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the global model on this client's local test partition."""

    client_id  = context.node_config["partition-id"]
    run_config = context.run_config
    experiment = run_config["experiment"]

    import importlib
    mltask = importlib.import_module(f"task.{experiment}")

    state_dict = msg.content["arrays"].to_torch_state_dict()
    model      = mltask.get_model()
    model.load_state_dict(state_dict)

    testloader = mltask.load_data(
        client_id,
        num_clients=run_config["num-clients"],
        num_workers=4,
        batch_size=run_config["batch-size"],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss, accuracy = mltask.test(model, testloader, device)

    print(
        f"[client {client_id}] eval — "
        f"accuracy: {round(100 * accuracy, 2)}%  loss: {round(loss, 4)}"
    )

    content = RecordDict({"metrics": MetricRecord({
        "loss":         loss,
        "accuracy":     accuracy,
        "num-examples": 1,
    })})
    return Message(content=content, reply_to=msg)
