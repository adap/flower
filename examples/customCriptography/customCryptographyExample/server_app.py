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
# FUNZIONE DI VALUTAZIONE SERVER-SIDE (secondo lo schema ufficiale Flower)
# ------------------------------
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple
import torch
from flwr.common import parameters_to_weights

def set_model_params(model, parameters):
    weights = parameters_to_weights(parameters)
    state_dict = dict(zip(model.state_dict().keys(), weights))
    model.load_state_dict(state_dict, strict=True)

def get_evaluate_fn():
    """Restituisce una funzione che valuta il modello globale sul server, come negli esempi ufficiali Flower."""

    # Prepara il validation loader una sola volta
    val_loader = get_validation_data(batch_size=64)
    criterion = torch.nn.CrossEntropyLoss()

    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Valutazione centralizzata del modello globale sul server."""

        print(f"\n🔹 Server-side evaluation - Round {server_round}")

        # Carica modello e aggiorna pesi con i parametri globali ricevuti
        model = get_model(NET, num_classes=10, pretrained=False)
        set_model_params(model, parameters)
        ndarrays = parameters_to_ndarrays(parameters)

        # Applica i pesi ricevuti
        state_dict = model.state_dict()
        for (k, v), new_p in zip(state_dict.items(), ndarrays):
            state_dict[k] = torch.tensor(new_p, dtype=v.dtype)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        total_loss, correct, total = 0.0, 0, 0

        # Loop di valutazione (uguale al reference Flower)
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Converte input nel formato corretto
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

        print(f"[Round {server_round}] Server-side accuracy: {accuracy:.4f} | loss: {avg_loss:.4f}")

        # ⬅️ Ritorna (loss, metrics) come richiede Flower
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
        fraction_evaluate=0.0,
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(),
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
