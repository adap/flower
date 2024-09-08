"""whisper_example: A Flower / PyTorch app with OpenAi's Whisper."""

from logging import INFO
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from whisper_example.dataset import get_encoding_fn
from whisper_example.model import eval_model, get_model, get_params, set_params

from datasets import load_dataset
from flwr.common import Context, FitRes, Metrics, NDArrays, ndarrays_to_parameters
from flwr.common.logger import log
from flwr.common.typing import UserConfig
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def get_evaluate_fn(
    val_set, test_set, processor: WhisperProcessor, run_config: UserConfig
):
    """Return a callback that the strategy will call after models are aggregated."""

    def evaluate(server_round: int, parameters: NDArrays, config):
        """Evaluate global model on a centralized dataset."""

        num_rounds = run_config["num-server-rounds"]
        num_classes = run_config["num-classes"]
        remove_cols = run_config["remove-cols"]
        remove_cols = remove_cols.split(",")

        # Determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # prepare model
        encoder, classifier = get_model(device, num_classes)
        set_params(classifier, parameters)
        classifier.to(device)

        # prepare dataset
        og_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        encoding_fn = get_encoding_fn(processor)
        if server_round == num_rounds:
            prefix = "test"
            encoded = test_set.map(encoding_fn, num_proc=4, remove_columns=remove_cols)
        else:
            prefix = "val"
            encoded = val_set.map(encoding_fn, num_proc=4, remove_columns=remove_cols)

        torch.set_num_threads(og_threads)
        val_encoded = encoded.with_format("torch", columns=["data", "targets"])
        val_loader = DataLoader(val_encoded, batch_size=64, num_workers=4)

        # Run global evaluation
        criterion = torch.nn.CrossEntropyLoss()
        loss, accuracy = eval_model(encoder, classifier, criterion, val_loader, device)

        return loss, {f"{prefix}_accuracy": accuracy}

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [
        num_examples * m["accuracy"] for num_examples, m in metrics if m["trained"]
    ]
    losses = [num_examples * m["loss"] for num_examples, m in metrics if m["trained"]]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_accuracy": sum(accuracies) / sum(examples),
        "train_loss": sum(losses) / sum(examples),
    }


class ExclusiveFedAvg(FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy | FitRes]],
        failures: List[Tuple[ClientProxy | FitRes] | BaseException],
    ):
        # Clients with not enough training examples to have a single full batch
        # didn't train the classification head. We need to exclude it from aggregation

        trained_results = []
        for cp, res in results:
            if res.metrics["trained"]:
                trained_results.append((cp, res))
        log(
            INFO,
            f"{len(trained_results)}/{len(results)} models included for aggregation.",
        )

        return super().aggregate_fit(server_round, trained_results, failures)


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    num_classes = context.run_config["num-classes"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize global model parameters. Recall we are
    # only federating the classification head
    _, classifier = get_model("cpu", num_classes, False)
    ndarrays = get_params(classifier)
    parameters = ndarrays_to_parameters(ndarrays)

    eval_fn = None
    if context.run_config["central-eval"]:
        # The ServerApp will use the validation set to assess the performance of the global
        # model after each round. Then, the test set will be used for evaluating the global
        # model after the last round
        sc_val = load_dataset(
            "speech_commands", "v0.02", split="validation", token=False
        )
        sc_test = load_dataset("speech_commands", "v0.02", split="test", token=False)

        # Processor
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

        # Prepare evaluation function
        eval_fn = get_evaluate_fn(
            val_set=sc_val,
            test_set=sc_test,
            processor=processor,
            run_config=context.run_config,
        )

    # Define the strategy
    strategy = ExclusiveFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_fn=eval_fn,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
