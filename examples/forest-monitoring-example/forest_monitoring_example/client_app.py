"""forest-monitoring-example: A Flower / PyTorch app."""

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from forest_monitoring_example.task import set_seed, load_model, get_weights, load_data_demo_npz, set_weights, test, train_FedAvg

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, 
                 net,
                 trainloader, valloader, target_scaler, feature_scaler,
                 local_epochs, initial_learning_rate, weight_decay,
                 id):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.target_scaler = target_scaler
        self.feature_scaler = feature_scaler
        self.local_epochs = local_epochs
        self.lr = initial_learning_rate
        self.wd = weight_decay
        self.device = DEVICE
        self.id = id

    def fit(self, parameters, config):

        set_weights(self.net, parameters)

        set_seed(42)

        # FedAvg
        all_epoch_losses = train_FedAvg(
            self.net,
            self.trainloader,
            self.valloader,
            self.target_scaler,
            self.lr,
            self.wd,
            self.local_epochs,
            self.device,
        )

        train_losses = all_epoch_losses['train_loss']

        training_loss = train_losses[-1] if train_losses else float('inf')  # Handle case where no training happens

        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"loss": training_loss,
             "client_id": self.id},
        )

    def evaluate(self, parameters, config):
        
        set_weights(self.net, parameters)

        loss_val, _, _, _, y_true, y_pred_global  = test(self.net, self.valloader, self.target_scaler, self.device)

        res = y_true - y_pred_global
        sse = float(np.sum(res**2))
        sum_y = float(np.sum(y_true))
        sum_y2 = float(np.sum(y_true**2))
        sum_pred = float(np.sum(y_pred_global))
        
        return float(loss_val), len(self.valloader.dataset), {"sse": sse, "sum_y": sum_y, "sum_y2": sum_y2, "sum_pred": sum_pred}


def client_fn(context: Context):

    # Load model and data
    if "partition-id" in context.node_config and "num-partitions" in context.node_config:
        print("Simulation mode")
        processed_data_npz = context.run_config["sim-data"]
        id = "CID"
    else:
        print("Deployment mode")
        processed_data_npz = context.node_config["processed-data-npz"]
        id = context.node_config["client-id"]

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
    initial_learning_rate = context.run_config["lr"]
    weight_decay = context.run_config["weight-decay"]

    trainloader, valloader, _, target_scaler, feature_scaler = load_data_demo_npz(
        processed_data_npz, batch_size,
        )
    
    net = load_model(feature_size, t_years, out_conv1, out_conv2, kernel_time, pool_time1, dropout_conv, adaptive_pool_time, use_adaptive_pool)

    # Return Client instance
    return FlowerClient(net,
                        trainloader, valloader, target_scaler, feature_scaler, 
                        local_epochs, initial_learning_rate, weight_decay,
                        id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
