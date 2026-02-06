"""flowertune-llm: A Flower / FlowerTune app."""

from __future__ import annotations

import os
from time import perf_counter
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig

from flowertune_llm.dataset import replace_keys
from flowertune_llm.models import get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Cache model state per node (outside Context.state to avoid RecordDict restrictions)
_MODEL_STATE: dict[int, dict[str, object]] = {}
_LAYER_NAMES: dict[int, list[str]] = {}


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Prepare model state for layer-wise sending (training disabled)."""
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    aggregation_mode = getattr(cfg, "aggregation", {}).get("mode", "layerwise")

    t0 = perf_counter()
    model = get_model(cfg.model)
    t1 = perf_counter()

    if msg.content and "arrays" in msg.content:
        model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=False)
    t2 = perf_counter()

    state_dict = model.state_dict()
    layer_names = list(state_dict.keys())
    if msg.content and "config" in msg.content:
        config = msg.content["config"]
        if "layer_names" in config:
            layer_names = list(config["layer_names"])

    _MODEL_STATE[context.node_id] = state_dict
    _LAYER_NAMES[context.node_id] = layer_names
    t3 = perf_counter()

    metrics = {
        "train_loss": 0.0,
        "num-examples": 1,
        "profile.client.prepare.ms": (t3 - t0) * 1000.0,
        "profile.client.load_model.ms": (t1 - t0) * 1000.0,
        "profile.client.load_arrays.ms": (t2 - t1) * 1000.0,
        "profile.client.state_dict.ms": (t3 - t2) * 1000.0,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})

    if aggregation_mode == "all_at_once":
        model_record = ArrayRecord(state_dict)
        content["arrays"] = model_record
    return Message(content=content, reply_to=msg)


@app.train("layer_wise_communication")
def train_comms(msg: Message, context: Context):
    """Send the model layer by layer."""
    if context.node_id not in _MODEL_STATE or context.node_id not in _LAYER_NAMES:
        cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
        model = get_model(cfg.model)
        state_dict = model.state_dict()
        layer_names = list(state_dict.keys())
        if msg.content and "config" in msg.content and "layer_names" in msg.content["config"]:
            layer_names = list(msg.content["config"]["layer_names"])
        _MODEL_STATE[context.node_id] = state_dict
        _LAYER_NAMES[context.node_id] = layer_names

    model_state = _MODEL_STATE[context.node_id]
    layer_names = _LAYER_NAMES[context.node_id]

    config = msg.content["config"] if msg.content and "config" in msg.content else {}
    layer_idx = int(config.get("layer_idx", 0))
    chunk_start = int(config.get("chunk_start", 0))
    chunk_end = int(config.get("chunk_end", 0))

    layer_name = layer_names[layer_idx]
    tensor = model_state[layer_name]
    t0 = perf_counter()
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "__getitem__") and chunk_end > chunk_start and getattr(
        tensor, "ndim", 0
    ):
        tensor = tensor[chunk_start:chunk_end]
    t1 = perf_counter()

    model_record = ArrayRecord({layer_name: tensor})
    t2 = perf_counter()

    metrics = {
        "num-examples": 1,
        "profile.client.slice.ms": (t1 - t0) * 1000.0,
        "profile.client.serialize.ms": (t2 - t1) * 1000.0,
    }

    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
