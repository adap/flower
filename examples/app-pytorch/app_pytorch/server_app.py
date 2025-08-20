"""app-pytorch: A Flower / PyTorch app."""

import torch
from app_pytorch.task import Net, test
from flwr.common import ArrayRecord, ConfigRecord, Context
from flwr.common.record.metricrecord import MetricRecord
from flwr.server import Grid, ServerApp
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_dataset

from .fedavg import FedAvg

# Create ServerApp
app = ServerApp()


def central_evaluation(server_round, array_record: ArrayRecord) -> MetricRecord:
    # Perform central evaluation

    # Instantiate model
    model = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Apply global model weights from message
    model.load_state_dict(array_record.to_torch_state_dict())
    model.to(device)

    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    dataset = test_dataset.with_format("torch", device=device).with_transform(
        apply_transforms
    )

    test_dataloader = DataLoader(dataset, batch_size=32)

    test_loss, test_acc = test(
        model,
        test_dataloader,
        device,
    )

    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


@app.main()
def main(grid: Grid, context: Context) -> None:

    # Init global model
    global_model = Net()

    # Init strategy
    strategy = FedAvg(fraction_train=context.run_config["fraction-train"])

    # Execute strategy loop
    num_rounds = context.run_config["num-server-rounds"]
    strategy_results = strategy.start(
        arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord({"lr": 0.01}),
        evaluate_config=ConfigRecord(
            {"num-batches": 10, "save-model-checkpoint": True}
        ),
        grid=grid,
        num_rounds=num_rounds,
        timeout=3600,
        central_eval_fn=central_evaluation,
    )

    print(strategy_results.train_metrics)
    print(strategy_results.evaluate_metrics)
    print(strategy_results.central_evaluate_metrics)
