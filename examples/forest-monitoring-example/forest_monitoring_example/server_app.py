"""forest-monitoring-example: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, MetricRecord, RecordDict, ConfigRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
import math

from forest_monitoring_example.task import load_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Create ServerApp
app = ServerApp()


def _aggregate_train_metrics(records: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Aggregate training metrics returned by clients in `@app.train()`."""
    total_w = 0
    weighted_loss = 0.0

    for record in records:
        m: MetricRecord = record["metrics"]
        w = int(m[weighting_key])
        total_w += w
        weighted_loss += w * float(m.get("train_loss", float("nan")))

    if total_w == 0:
        return MetricRecord({})

    return MetricRecord({"train_loss": weighted_loss / total_w})


def _aggregate_eval_metrics_from_sums(records: list[RecordDict], weighting_key: str) -> MetricRecord:
    """Aggregate client evaluation sums into global metrics.

    Clients should return SSE and sufficient sums to compute R²:
      - sse = Σ (y - ŷ)^2
      - sum_y = Σ y
      - sum_y2 = Σ y^2
      - sum_pred = Σ ŷ   (used for RMSE% denominator)
    """
    n_total = 0
    sse_total = 0.0
    sum_y_total = 0.0
    sum_y2_total = 0.0
    sum_pred_total = 0.0
    # Optional: weighted average of eval loss (e.g. MSE in scaled space)
    eval_loss_total = 0.0

    for record in records:
        m: MetricRecord = record["metrics"]
        n = int(m[weighting_key])

        n_total += n
        sse_total += float(m.get("sse", 0.0))
        sum_y_total += float(m.get("sum_y", 0.0))
        sum_y2_total += float(m.get("sum_y2", 0.0))
        sum_pred_total += float(m.get("sum_pred", 0.0))
        eval_loss_total += n * float(m.get("eval_loss", 0.0))

    if n_total == 0:
        return MetricRecord({})

    mse = sse_total / n_total
    rmse = math.sqrt(mse)

    # R² using totals
    sst = sum_y2_total - (sum_y_total**2) / n_total
    r2 = float("nan") if sst <= 0 else 1.0 - (sse_total / sst)

    mean_pred = sum_pred_total / n_total
    rmse_pct = float("nan") if mean_pred == 0.0 else 100.0 * rmse / mean_pred

    eval_loss = eval_loss_total / n_total

    return MetricRecord(
        {
            "rmse": rmse,
            "rmse%": rmse_pct,
            "r2": r2,
            "mse_orig": mse,
            "eval_loss": eval_loss,
        }
    )



@app.main()
def main(grid: Grid, context: Context) -> None:

    # Read from config
    # run config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_evaluate=context.run_config["fraction-evaluate"]
    save_final: bool = bool(context.run_config.get("save-final-model", False))

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

    

    # Initialize model parameters
    net = load_model(
        feature_size, 
        t_years, 
        out_conv1, 
        out_conv2, 
        kernel_time, 
        pool_time1, 
        dropout_conv, 
        adaptive_pool_time, 
        use_adaptive_pool,
    ).to(DEVICE)
    
    
    arrays = ArrayRecord(net.state_dict())

    # Optional: send extra config to clients each round (available in msg.content["config"])
    train_cfg = ConfigRecord(
        {
            "local-epochs": int(context.run_config.get("local-epochs", 1)),
            "lr": float(context.run_config.get("lr", 1e-3)),
            "weight-decay": float(context.run_config.get("weight-decay", 0.0)),
        }
    )
    eval_cfg = ConfigRecord({})

    # Define strategy
    strategy = FedAvg(
        fraction_evaluate=fraction_evaluate,
        train_metrics_aggr_fn=_aggregate_train_metrics,
        evaluate_metrics_aggr_fn=_aggregate_eval_metrics_from_sums,
        )

    # Start the strategy
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_cfg,
        evaluate_config=eval_cfg,
        num_rounds=num_rounds,
        evaluate_fn=None,
    )
    print(result)

    # Save final model to disk
    if save_final and result.arrays is not None:
        print("\nSaving final model to disk -> final_model.pt")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

