from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .task import get_weights, get_model, get_validation_data
from flwr.common.crypto.config_cripto import NET, ACCURACY

import torch


# ------------------------------
# METRICA AGGREGATA
# ------------------------------
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


# ------------------------------
# STRATEGIA CON VALUTAZIONE SERVER-SIDE
# ------------------------------
class FedAvgWithServerEval(FedAvg):
    """Estensione di FedAvg che esegue evaluate_fn lato server dopo ogni aggregazione."""

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None and self.evaluate_fn is not None:
            parameters, _ = aggregated
            print(f"\n🔹 Esecuzione valutazione server-side dopo round {server_round}...")
            self.evaluate_fn(server_round, parameters, {})

        return aggregated


# ------------------------------
# FUNZIONE DI VALUTAZIONE
# ------------------------------
def get_evaluate_fn():
    """Return a function to evaluate the global model on the server."""

    def evaluate(server_round: int, parameters, config):
        print(f"Server-side evaluation - Round {server_round}")

        # Convert parameters to numpy arrays
        ndarrays = parameters_to_ndarrays(parameters)

        # Load model and update weights
        model = get_model(NET, num_classes=10, pretrained=True)
        state_dict = model.state_dict()
        for (k, v), new_p in zip(state_dict.items(), ndarrays):
            try:
                state_dict[k] = torch.tensor(new_p, dtype=state_dict[k].dtype)
            except Exception:
                print(f"[WARN] Skipping incompatible layer: {k}, shape {new_p.shape}")
        model.load_state_dict(state_dict, strict=False)

        val_loader = get_validation_data(batch_size=64)
        criterion = torch.nn.CrossEntropyLoss()
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Evaluation loop
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Fix input format if needed
                if isinstance(inputs, list):
                    inputs = torch.stack(inputs)
                if inputs.ndim == 4 and inputs.shape[1] != 3:
                    inputs = inputs.permute(0, 3, 1, 2)
                inputs = inputs.to(torch.float32)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total += targets.size(0)

        accuracy = correct / total
        avg_loss = total_loss / total
        print(f"[Round {server_round}] Server-side accuracy: {accuracy:.4f} | loss: {avg_loss:.4f}")
        return avg_loss, {"accuracy": accuracy}

    return evaluate


# ------------------------------
# SERVER APP
# ------------------------------
def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    net = get_model(NET, num_classes=10, pretrained=True)
    parameters = ndarrays_to_parameters(get_weights(net))

    strategy = FedAvgWithServerEval(
        fraction_fit=0.2,
        fraction_evaluate=0.0,  # solo valutazione server-side
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),
        initial_parameters=parameters,
        stop_criteria={"metric_ge": ("accuracy", ACCURACY)},
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# ------------------------------
# AVVIO SERVER
# ------------------------------
app = ServerApp(server_fn=server_fn)
