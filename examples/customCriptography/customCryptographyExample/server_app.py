from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flwr.common.crypto.config_cripto import NET, ACCURACY
from flwr.common import NDArrays, Scalar
from typing import  OrderedDict
from collections import OrderedDict
import torch
from flwr.common import Context, NDArrays, Scalar, parameters_to_ndarrays

from flwr.common.crypto.log_file import log_time

from .task import get_model, get_weights, set_weights, test, get_validation_data
from flwr.common.crypto.config_cripto import NET,EVALUATION_SIDE  # o "resnet18" se preferisci esplicitarlo

from flwr.server.history import History
from flwr.common.crypto.config_cripto import ACCURACY


# ------------------------------
# METRICA AGGREGATA
# ------------------------------
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


class FedAvgWithServerEval(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history: History | None = None
        self.last_server_eval = None  # ⬅️ salva l'ultimo risultato server-side

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None:
            print(f"[Round {server_round}] Nessun aggregato disponibile (results vuoto o fallimenti). Skip server eval.")
            return aggregated

        parameters, _ = aggregated
        if self.evaluate_fn is None:
            return aggregated

        if parameters is None or getattr(parameters, "tensors", None) in (None, [], ()):
            print(f"[Round {server_round}] Parameters=None/empty dopo aggregazione. Skip server eval.")
            return aggregated

        try:
            print(f"\n🔹 Esecuzione valutazione server-side dopo round {server_round}...")
            result = self.evaluate_fn(server_round, parameters)

            if result is not None:
                loss, metrics = result
                accuracy = metrics.get("accuracy", 0.0)
                print(f"✅ [Round {server_round}] Server-side accuracy: {accuracy:.4f} | loss: {loss:.4f}")

                # ⬇️ salva per il server
                self.last_server_eval = (loss, metrics)

                if self.history is not None:
                    self.history.add_loss_centralized(server_round, loss)
                    self.history.add_metrics_centralized(server_round, metrics)

                if accuracy >= ACCURACY:
                    print(f"🎯 Accuratezza target ({ACCURACY:.4f}) raggiunta. Arresto controllato del training.")
                    self.stop_triggered = True

        except Exception as e:
            print(f"[Round {server_round}] Server eval saltata per errore: {e}")

        return aggregated

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def server_evaluate(server_round: int, parameters: NDArrays):
    """Valuta il modello globale ResNet18 sul test set CIFAR-10 (server-side evaluation)."""
    print(f"\n🔹 Server-side evaluation - Round {server_round}")

    # ✅ Aggiungi questa guardia:
    if parameters is None or getattr(parameters, "tensors", None) in (None, [], ()):
        print(f"[Round {server_round}] Parameters=None/empty: salto valutazione server-side.")
        return None

    model = get_model(NET, num_classes=10, pretrained=True).to(DEVICE)
    ndarrays = parameters_to_ndarrays(parameters)
    params_dict = zip(model.state_dict().keys(), ndarrays)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    testloader = get_validation_data(batch_size=64)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0

    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"✅ [Round {server_round}] Server-side accuracy: {accuracy:.4f} | loss: {avg_loss:.4f}")
    return avg_loss, {"accuracy": accuracy}


# ------------------------------
# SERVER APP
# ------------------------------
def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    net = get_model(NET, num_classes=10, pretrained=False)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)
    if EVALUATION_SIDE == "server":
        server_side = server_evaluate
        fraction_evaluate=0.0
    else:
        server_side = None
        fraction_evaluate=context.run_config["fraction-evaluate"]


    log_time(f"local-epochs:  {context.run_config['local-epochs']}")
    log_time(f"learning-rate:  {context.run_config['learning-rate']}")
    log_time(f"batch-size:  {context.run_config['batch-size']}" )
    log_time(f"fraction-evaluate:  {context.run_config['fraction-evaluate']}" )

    if server_side is not None:
        log_time("SERVER SIDE EVALUATION")
    else:
        log_time("CLIENT SIDE EVALUATION")

    strategy = FedAvgWithServerEval(
        fraction_fit=context.run_config["fraction-fit"],
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        evaluate_fn=server_side,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        stop_criteria={"metric_ge": ("accuracy", ACCURACY),
        },
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)
# ------------------------------
# AVVIO SERVER
# ------------------------------
app = ServerApp(server_fn=server_fn)
