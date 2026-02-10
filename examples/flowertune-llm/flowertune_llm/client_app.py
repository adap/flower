"""flowertune-llm: A Flower / FlowerTune app."""

from __future__ import annotations

import os
import pickle
import re
import warnings

from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

STATE_LAYER_NAMES = "layer_names"
STATE_LAYER_PATHS = "layer_paths"
STATE_LAYER_IDX = "layer_idx"
STATE_NUM_EXAMPLES = "num_examples"


# Flower ClientApp
app = ClientApp()


def _sanitize_layer_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


@app.train()
def train(msg: Message, context: Context):
    """Prepare model state for layer-wise sending (training disabled)."""
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    aggregation_mode = getattr(cfg, "aggregation", {}).get("mode", "layerwise")

    # Load model (no training)
    model = get_model(cfg.model)

    if msg.content and "arrays" in msg.content:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    state_dict = model.state_dict()
    layer_names = list(state_dict.keys())
    if msg.content and "config" in msg.content:
        config = msg.content["config"]
        if "layer_names" in config:
            layer_names = list(config["layer_names"])

    # Persist layer names in context (lightweight)
    context.state[STATE_LAYER_NAMES] = ConfigRecord({"names": layer_names})

    # Persist layers to disk for per-layer sending
    layer_dir = os.path.join(os.getcwd(), "layers", str(context.run_id), str(context.node_id))
    os.makedirs(layer_dir, exist_ok=True)

    serialized_layer_paths: list[str] = []
    for layer_name in layer_names:
        file_name = f"{_sanitize_layer_name(layer_name)}.pt"
        file_path = os.path.join(layer_dir, file_name)
        serialized_layer_paths.append(file_path)
        with open(file_path, "wb") as file:
            pickle.dump({layer_name: state_dict[layer_name]}, file)

    # Tracking state for layer-wise communication
    context.state[STATE_LAYER_PATHS] = ConfigRecord({"paths": serialized_layer_paths})
    context.state[STATE_LAYER_IDX] = ConfigRecord({"idx": 0})
    context.state[STATE_NUM_EXAMPLES] = ConfigRecord({"num_examples": 1})

    metrics = {
        "train_loss": 0.0,
        "num-examples": 1,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": ArrayRecord(), "metrics": metric_record})

    if aggregation_mode == "all_at_once":
        content["arrays"] = ArrayRecord(state_dict)

    return Message(content=content, reply_to=msg)


@app.train("layer_wise_communication")
def train_comms(msg: Message, context: Context):
    """Send the model layer by layer from disk."""
    layer_idx = int(context.state[STATE_LAYER_IDX]["idx"])
    layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])

    if layer_idx >= len(layer_paths):
        layer_idx = len(layer_paths) - 1

    send_complete = layer_idx >= (len(layer_paths) - 1)

    # Read model layer from disk
    layer_path = layer_paths[layer_idx]
    with open(layer_path, "rb") as file:
        layer_dict = pickle.load(file)

    layer_name = next(iter(layer_dict.keys()))
    array_record = ArrayRecord(layer_dict)

    num_examples = int(context.state[STATE_NUM_EXAMPLES]["num_examples"])
    metric_record = MetricRecord({"num-examples": num_examples})

    config_record = ConfigRecord({"send_complete": send_complete})
    content = RecordDict({
        "arrays": array_record,
        "metrics": metric_record,
        "config": config_record,
    })

    context.state[STATE_LAYER_IDX] = ConfigRecord({"idx": layer_idx + 1})
    return Message(content=content, reply_to=msg)
