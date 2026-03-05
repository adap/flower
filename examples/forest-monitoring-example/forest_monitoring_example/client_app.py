"""Flower ClientApp (new API) for forest_monitoring_example.

Implements:
- @app.train(): receives global model arrays, trains locally, returns updated arrays + training metrics
- @app.evaluate(): receives global model arrays, evaluates locally, returns evaluation metrics

This ClientApp supports both:
- Simulation mode (when context.node_config has partition info)
- Deployment mode (when context.node_config contains paths/ids per client)
"""

import numpy as np
from logging import INFO
import torch
from flwr.client import ClientApp
from flwr.common import Context, log

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from forest_monitoring_example.task import set_seed, load_model, load_data_demo_npz, test, train_FedAvg

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Flower ClientApp
app = ClientApp()

def _is_simulation(context: Context) -> bool:
    return ("partition-id" in context.node_config) and ("num-partitions" in context.node_config)

def _get_client_id(context: Context) -> str:
    if _is_simulation(context):
        pid = context.node_config.get("partition-id")
        return f"partition-{pid}"
    return str(context.node_config.get("client-id", "client"))

def _get_npz_path(context: Context) -> str:
    if _is_simulation(context):
        return str(context.run_config["sim-data"])
    return str(context.node_config["processed-data-npz"])



@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the model on local data."""
    
    set_seed(int(context.run_config.get("seed", 42)))

    client_id = _get_client_id(context)
    log(INFO, "Train on client id %s", _get_client_id(context))
    npz_path = _get_npz_path(context)

    batch_size = context.run_config["batch-size"]
    local_epochs = context.run_config["local-epochs"]

    # model config:
    feature_size = context.run_config["feature-size"]
    t_years = context.run_config["t-years"]
    out_conv1 = context.run_config["out-conv1"]
    out_conv2 = context.run_config["out-conv2"]
    kernel_time = context.run_config["kernel-time"]
    pool_time1 = context.run_config["pool-time1"]
    dropout_conv = context.run_config["dropout-conv"]
    adaptive_pool_time = context.run_config["adaptive-pool-time"]
    use_adaptive_pool = context.run_config["use-adaptive-pool"]
    
    # train config:
    lr = context.run_config["lr"]
    wd = context.run_config["weight-decay"]

    trainloader, valloader, _testloader, target_scaler, _feature_scaler = load_data_demo_npz(
        npz_path, batch_size,
        )
    
    net = load_model(
        feature_size, 
        t_years, 
        out_conv1,
        out_conv2, 
        kernel_time, 
        pool_time1, 
        dropout_conv, 
        adaptive_pool_time, 
        use_adaptive_pool
    ).to(DEVICE)
    
    # Read ArrayRecord received from ServerApp
    arrays: ArrayRecord = msg.content["arrays"]
    
    # Load weights to model
    net.load_state_dict(arrays.to_torch_state_dict())

    # FedAvg
    all_epoch_losses = train_FedAvg(
        net,
        trainloader,
        valloader,
        target_scaler,
        lr,
        wd,
        local_epochs,
        DEVICE,
    )

    #train_losses = all_epoch_losses['train_loss']
    train_losses = all_epoch_losses.get("train_loss", [])
    training_loss = train_losses[-1] if train_losses else float('inf')  # Handle case where no training happens

    # Construct reply Message: arrays and metrics
    model_record = ArrayRecord(net.state_dict())
    # You can include any metric (scalar or list of scalars)
    # relevant to your usecase.
    # A weighting metric (`num-examples` by default) is always
    # expected by FedAvg to do aggregation
    metrics = MetricRecord(
        {
            "train_loss": training_loss,
            "num-examples": len(trainloader.dataset),
            #"client_id": id
        }
    )
    # Construct RecordDict and add ArrayRecord and MetricRecord
    content = RecordDict({"arrays": model_record, "metrics": metrics})
    
    return Message(content=content, reply_to=msg)
    


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the model on local data."""

    log(INFO, "EVALUATE called on client %s", _get_client_id(context))

    set_seed(int(context.run_config.get("seed", 42)))
    
    # Load model and data
    npz_path = _get_npz_path(context)
    log(INFO, "npz-file is %s", npz_path)
    batch_size = context.run_config["batch-size"]

    # model config:
    feature_size = context.run_config["feature-size"]
    t_years = context.run_config["t-years"]
    out_conv1 = context.run_config["out-conv1"]
    out_conv2 = context.run_config["out-conv2"]
    kernel_time = context.run_config["kernel-time"]
    pool_time1 = context.run_config["pool-time1"]
    dropout_conv = context.run_config["dropout-conv"]
    adaptive_pool_time = context.run_config["adaptive-pool-time"]
    use_adaptive_pool = context.run_config["use-adaptive-pool"]
    

    _, valloader, _, target_scaler, _ = load_data_demo_npz(
        npz_path, batch_size,
        )
    
    net = load_model(
        feature_size, 
        t_years, 
        out_conv1, 
        out_conv2, 
        kernel_time, 
        pool_time1, 
        dropout_conv, 
        adaptive_pool_time, 
        use_adaptive_pool
    ).to(DEVICE)
    
    # Read ArrayRecord received from ServerApp
    arrays: ArrayRecord = msg.content["arrays"]
    
    # Load weights to model
    net.load_state_dict(arrays.to_torch_state_dict())

    loss_val, _, _, _, y_true, y_pred_global  = test(net, valloader, target_scaler, DEVICE)

    res = y_true - y_pred_global
    sse = float(np.sum(res**2))
    sum_y = float(np.sum(y_true))
    sum_y2 = float(np.sum(y_true**2))
    sum_pred = float(np.sum(y_pred_global))

    # Construct reply Message
    # Retrun metrics relevant to usecase
    # THe weighting metric is also sent and will be used
    # to do weighted aggregation of metrics
    metrics = MetricRecord(
        {
            "eval_loss": float(loss_val),
            "sse": sse, 
            "sum_y": sum_y, 
            "sum_y2": sum_y2, 
            "sum_pred": sum_pred,
            "num-examples": len(valloader.dataset),
        }
    )
    # Construct RecordDict and add MetricRecord
    content = RecordDict({"metrics": metrics})
    return Message(content=content, reply_to=msg)

