"""forest-monitoring-example: A Flower / PyTorch app."""

from collections import OrderedDict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy, math, random


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class EarlyStopping:
    """
    Early stops training if validation MSE doesn't improve enough for `patience` epochs.
    You pass validation MSE to __call__(val_mse).
    """
    def __init__(self, patience=20, verbose=False,
                 min_delta_rel=0.01,      # e.g., 0.01 = require 1% MSE improvement
                 min_delta_abs=0.0,       # optional absolute floor in (m³/ha)²
                 trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best = float('inf')          # best validation MSE so far
        self.early_stop = False
        self.min_delta_rel = float(min_delta_rel)
        self.min_delta_abs = float(min_delta_abs)
        self.trace_func = trace_func

    def __call__(self, val_mse: float):
        # Relative threshold w.r.t. the current best
        rel_thresh = self.min_delta_rel * self.best if self.best < float('inf') else 0.0
        delta = max(self.min_delta_abs, rel_thresh)  # require at least this much improvement

        improved = val_mse < (self.best - delta)
        if improved:
            self.best = val_mse
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"ES: {self.counter}/{self.patience} "
                    f"(best MSE={self.best:.2f}, current MSE={val_mse:.2f}, "
                    f"required Δ≥{delta:.2f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True


class TimeSeriesCNN(nn.Module):
    """ Time Series CNN model"""
    def __init__(self, 
                 n_features, t_years, 
                 out_conv1, out_conv2,
                 kernel_time, pool_time1, dropout_conv, 
                 adaptive_pool_time, use_adaptive_pool=True):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=out_conv1,
            kernel_size=(n_features, kernel_time),
            stride=1,
            padding=(0, kernel_time // 2)
        )
        if pool_time1 > 1:
            self.pool1 = nn.MaxPool2d(kernel_size=(1, pool_time1), stride=pool_time1)
        else:
            self.pool1 = nn.Identity()
        self.conv2 = nn.Conv2d(
            in_channels=out_conv1,
            out_channels=out_conv2,
            kernel_size=(1, kernel_time),
            stride=1,
            padding=(0, kernel_time // 2)
        )

        if use_adaptive_pool:
            self.pool2 = nn.AdaptiveAvgPool2d((1, adaptive_pool_time))
            fc_input_size = out_conv2 * adaptive_pool_time
        else:
            self.pool2 = nn.Identity()
            # compute output temporal size after conv2 manually
            def conv1d_out_length(L_in, kernel, stride=1, padding=0):
                return (L_in + 2*padding - kernel) // stride + 1

            # After conv1
            t_after_conv1 = conv1d_out_length(t_years, kernel_time, stride=1, padding=kernel_time // 2)
            # After pool1
            t_after_pool1 = t_after_conv1 if pool_time1 <= 1 else t_after_conv1 // pool_time1
            # After conv2
            t_after_conv2 = conv1d_out_length(t_after_pool1, kernel_time, stride=1, padding=kernel_time // 2)

            fc_input_size = out_conv2 * t_after_conv2

        self.dropout_conv = nn.Dropout(dropout_conv)
        self.fc = nn.Linear(fc_input_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze()
    

def load_model(feature_size, t_years, out_conv1, out_conv2, kernel_time, pool_time1, dropout_conv, adaptive_pool_time, use_adaptive_pool):
    return TimeSeriesCNN(n_features=feature_size,
                        t_years=t_years,
                        out_conv1 = out_conv1, 
                        out_conv2 = out_conv2,
                        kernel_time=kernel_time,
                        pool_time1=pool_time1,
                        dropout_conv=dropout_conv,
                        adaptive_pool_time = adaptive_pool_time, 
                        use_adaptive_pool = use_adaptive_pool
            )

def _make_standard_scaler_from_params(mean: np.ndarray, std: np.ndarray) -> StandardScaler:
    # Convert to 1-D arrays regardless of input shape (scalar -> length-1)
    mean = np.asarray(mean, dtype=np.float64).reshape(-1)
    std = np.asarray(std, dtype=np.float64).reshape(-1)

    if mean.shape != std.shape:
        raise ValueError(f"mean and std must have same shape, got {mean.shape} vs {std.shape}")
    
    safe_std = np.where(std == 0.0, 1.0, std)

    sc = StandardScaler()
    sc.mean_ = mean
    sc.scale_ = safe_std
    sc.var_ = safe_std ** 2
    sc.n_features_in_ = mean.size  # 1 for scalar target, f for feature vector
    
    return sc


def load_data_demo_npz(
    npz_path: str,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    data = np.load(npz_path, allow_pickle=True)

    def make_loader(X_key, y_key, shuffle):
        X = torch.from_numpy(data[X_key]).float()
        y = torch.from_numpy(data[y_key]).float()
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    trainloader = make_loader("X_train", "y_train", shuffle=True)
    valloader   = make_loader("X_val",   "y_val",   shuffle=False)
    testloader  = None
    if "X_test" in data.files and "y_test" in data.files:
        testloader = make_loader("X_test", "y_test", shuffle=False)

    # Reconstruct scalers if present (optional)
    feature_scaler = None
    target_scaler  = None
    if "feature_mean" in data.files and "feature_std" in data.files:
        feature_scaler = _make_standard_scaler_from_params(data["feature_mean"], data["feature_std"])
    if "target_mean" in data.files and "target_std" in data.files:
        target_scaler = _make_standard_scaler_from_params(data["target_mean"], data["target_std"])

    return trainloader, valloader, testloader, target_scaler, feature_scaler



# FedAvg
def train_FedAvg(net, trainloader, valloader, target_scaler, lr, wd, epochs, device):
    
    net = net.to(device)

    criterion = torch.nn.MSELoss() # No need to move criterion to device in modern PyTorch
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)  
    # Monitor validation loss for early stopping
    early_stopping = EarlyStopping(patience=3, min_delta_rel=0.001, min_delta_abs=0.0, verbose=False) 
    best_state, best_mse = None, math.inf

    history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

    for epoch in range(epochs):
        net.train() # Set model to training mode
        running_train_loss = 0.0
        
        for batch in trainloader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)

            # Ensure outputs and targets have the same 2D structure: If output is 1D add a batch dimension
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            if targets.dim() == 0: targets = targets.unsqueeze(0)

            loss_batch = criterion(outputs, targets)
            
            loss_batch.backward()
            optimizer.step()
            
            # Accumulate training loss per batch
            running_train_loss += loss_batch.item() * inputs.size(0) # Weighted by batch size

        # Calculate average training loss for the epoch
        avg_train_loss = running_train_loss / len(trainloader.dataset)
        history['train_loss'].append(avg_train_loss)

        # --- Validation Step ---
        # Get validation metrics from your 'test' function
        val_loss, val_rmse, _, _, _, _ = test(net, valloader, target_scaler, device)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")

        if val_loss < best_mse:
            best_mse = val_loss
            best_state = copy.deepcopy(net.state_dict())

        # Use the validation loss for early stopping check
        early_stopping(val_loss) 
        if early_stopping.early_stop:
            print(f"🛑 Early stopping triggered at epoch {epoch+1}")
            break
    
    # restore best
    if best_state is not None:
        net.load_state_dict(best_state)

    sigma = float(target_scaler.scale_[0])
    best_mse_orig  = math.sqrt(best_mse) * sigma
    print(f"Best validation RMSE in orig scale: {best_mse_orig:.2f} m³/ha")

    #return history # Return a dictionary of history for better plotting/analysis
    return history


def calc_metrics(actuals, predictions):
    # Convert inputs to numpy arrays (ensures calculations work correctly)
    actuals = np.array(actuals)
    predictions = np.array(predictions)
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    # RMSE as percentage of mean observations
    rmse_pct = 100 * rmse / np.mean(actuals)
    # Calculate R2
    r_squared = r2_score(actuals, predictions)
    # Calculate Bias (Mean Error)
    bias = np.mean(predictions - actuals)

    return rmse, rmse_pct.item(), r_squared, bias


def test(net, dataloader, target_scaler, device):
    net = net.to(device)
    criterion = torch.nn.MSELoss()

    net.eval()
    running_loss = 0.0
    predictions_list, actuals_list = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move inputs and targets to the device (GPU or CPU)
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = net(inputs)   #  .squeeze()

            # Optional edge case handling: 
            # Ensure outputs and targets have the same 2D structure: If output is 1D add a batch dimension
            if outputs.dim() == 0: outputs = outputs.unsqueeze(0)
            if targets.dim() == 0: targets = targets.unsqueeze(0)
            
            # Accumulate loss weighted by batch size
            batch_loss = criterion(outputs, targets).item()
            running_loss += batch_loss * inputs.size(0)

            # Collect raw tensor values efficiently
            predictions_list.extend(outputs.cpu().numpy())
            actuals_list.extend(targets.cpu().numpy())
    
    # Calculate average loss across all samples
    avg_loss = running_loss / len(dataloader.dataset)

    # --- Metrics Calculation ---
    # Convert lists to numpy arrays for inverse transform/metrics calculation
    # Reshape might be needed depending on the scaler implementation requirements
    predictions_np = np.array(predictions_list).reshape(-1, 1)
    actuals_np = np.array(actuals_list).reshape(-1, 1)

    # back-transform predictions
    # Inverse transform predictions and true values to original scale
    predictions_backsc = target_scaler.inverse_transform(predictions_np).flatten()
    actuals_backsc = target_scaler.inverse_transform(actuals_np).flatten()

    # calculate metrics
    rmse, rmse_pct, r_squared, _ = calc_metrics(actuals_backsc, predictions_backsc)
    
    # Note that avg_loss is in transformed scale, but rmse, rmse%, r2, actuals_backsc, predictions_backsc are in original (backtransformed) scale!
    return avg_loss, rmse, rmse_pct, r_squared, actuals_backsc, predictions_backsc


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
