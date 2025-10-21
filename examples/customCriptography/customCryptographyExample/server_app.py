from datasets import load_dataset
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flwr.common.crypto.config_cripto import NET, ACCURACY
from flwr.common import NDArrays, Scalar
from typing import  OrderedDict
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import torch
import numpy as np
from flwr.common import Context, NDArrays, Scalar, parameters_to_ndarrays

# Importa le tue utility
from .task import get_model, get_weights, set_weights, test, get_validation_data
from flwr.common.crypto.config_cripto import NET  # o "resnet18" se preferisci esplicitarlo



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
    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None:
            print(f"[Round {server_round}] Nessun aggregato disponibile (results vuoto o fallimenti). Skip server eval.")
            return aggregated

        parameters, _ = aggregated

        if self.evaluate_fn is None:
            return aggregated

        # 🔒 Evita chiamate con parameters=None o vuoti
        if parameters is None or getattr(parameters, "tensors", None) in (None, [], ()):
            print(f"[Round {server_round}] Parameters=None/empty dopo aggregazione. Skip server eval.")
            return aggregated

        try:
            print(f"\n🔹 Esecuzione valutazione server-side dopo round {server_round}...")
            self.evaluate_fn(server_round, parameters)
        except Exception as e:
            print(f"[Round {server_round}] Server eval saltata per errore: {e}")

        return aggregated



def get_evaluate_fn():
    """Restituisce una funzione che valuta il modello globale sul server."""

    # Prepara il validation loader una sola volta
    val_loader = get_validation_data(batch_size=64)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(
            server_round: int,
            parameters: NDArrays,
            config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Valutazione centralizzata del modello globale sul server."""

        print(f"\n🔹 Server-side evaluation - Round {server_round}")

        # Carica modello e aggiorna pesi con i parametri globali ricevuti
        model = get_model(NET, num_classes=10, pretrained=False)

        # 🔹 Conversione coerente con il client
        ndarrays = parameters_to_ndarrays(parameters)
        params_dict = zip(model.state_dict().keys(), ndarrays)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                # Normalizzazione e permutazione canali se necessario
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

        # Calcola metriche aggregate
        avg_loss = total_loss / total
        accuracy = correct / total

        print(f"✅ [Round {server_round}] Server-side accuracy: {accuracy:.4f} | loss: {avg_loss:.4f}")

        # ⬅️ Ritorna (loss, metrics)
        return avg_loss, {"accuracy": accuracy}

    return evaluate

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def server_evaluate(server_round: int, parameters: NDArrays):
    """Valuta il modello globale ResNet18 sul test set CIFAR-10 (server-side evaluation)."""
    print(f"\n🔹 Server-side evaluation - Round {server_round}")

    # ✅ Aggiungi questa guardia:
    if parameters is None or getattr(parameters, "tensors", None) in (None, [], ()):
        print(f"[Round {server_round}] Parameters=None/empty: salto valutazione server-side.")
        return None

    model = get_model("resnet18", num_classes=10, pretrained=False).to(DEVICE)
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
    return avg_loss, {"centralized_accuracy": accuracy}




# ------------------------------
# SERVER APP
# ------------------------------
def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    net = get_model(NET, num_classes=10, pretrained=True)
    parameters = ndarrays_to_parameters(get_weights(net))

    strategy = FedAvgWithServerEval(
        fraction_fit=0.2,
        fraction_evaluate=0.0,
        min_available_clients=1,
        evaluate_fn=server_evaluate,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        stop_criteria={"metric_ge": ("accuracy", ACCURACY)},
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)
# ------------------------------
# AVVIO SERVER
# ------------------------------
app = ServerApp(server_fn=server_fn)
