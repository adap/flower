"""app-pytorch: A Flower / PyTorch app."""

from pprint import pprint

import torch
from app_pytorch.task import Net, test
from flwr.common import ArrayRecord, ConfigRecord, Context
from flwr.common.record.metricrecord import MetricRecord
from flwr.server import Grid, ServerApp
from flwr.serverapp import FedAvg
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import load_dataset

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Load global model
    global_model = Net()

    # Init strategy
    strategy = FedAvg(
        fraction_train=context.run_config["fraction-train"],
    )

    # Execute strategy loop
    num_rounds = context.run_config["num-server-rounds"]
    strategy_results = strategy.start(
        grid=grid,
        arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord({"lr": 0.01}),
        num_rounds=num_rounds,
        timeout=3600,
        central_eval_fn=central_evaluation,
    )

    # Log resulting metrics
    print("\nTrain metrics:")
    pprint(strategy_results.train_metrics)
    print("\nDistributed evaluate metrics:")
    pprint(strategy_results.evaluate_metrics)
    print("\nCentral evaluate metrics:")
    pprint(strategy_results.central_evaluate_metrics)

    # Save final model to disk
    state_dict = strategy_results.arrays.to_torch_state_dict()
    print("\nSaving final model to disk.")
    torch.save(state_dict, "final_model.pt")


def central_evaluation(server_round, array_record: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(array_record.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
