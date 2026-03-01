from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
import logging
import csv
from flwr.common.logger import log
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from flwr.common.crypto.config_cripto import NET, ACCURACY
from flwr.common import NDArrays, Scalar
from typing import  OrderedDict
from collections import OrderedDict
from pathlib import Path
import torch
from flwr.common import Context, NDArrays, Scalar, parameters_to_ndarrays

from flwr.common.crypto.log_file import log_time

from .task import get_model, get_weights, set_weights, test, get_validation_data
from flwr.common.crypto.config_cripto import NET,EVALUATION_SIDE  # o "resnet18" se preferisci esplicitarlo

from flwr.server.history import History
from flwr.common.crypto.config_cripto import ACCURACY



CLIENT_METRICS_CSV = Path("logs/client_metriche_round.csv")


def _append_client_metric_row(
    round_id,
    client_idx,
    num_examples,
    tempo_cpu,
    tempo_reale,
    core_equivalenti,
    percentuale_cpu,
    ram_iniziale_mb,
    ram_finale_mb,
    delta_ram_mb,
    percentuale_ram_sistema,
    epoche_locali,
    learning_rate_fit,
):
    CLIENT_METRICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not CLIENT_METRICS_CSV.exists()

    with CLIENT_METRICS_CSV.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(
                [
                    "round",
                    "client_idx",
                    "esempi",
                    "tempo_cpu_fit",
                    "tempo_reale_fit",
                    "core_equivalenti_fit",
                    "percentuale_cpu_fit",
                    "ram_iniziale_mb_fit",
                    "ram_finale_mb_fit",
                    "delta_ram_mb_fit",
                    "percentuale_ram_sistema_fit",
                    "epoche_locali_fit",
                    "learning_rate_fit",
                ]
            )

        writer.writerow(
            [
                round_id,
                client_idx,
                num_examples,
                f"{tempo_cpu:.6f}",
                f"{tempo_reale:.6f}",
                f"{core_equivalenti:.6f}",
                f"{percentuale_cpu:.6f}",
                f"{ram_iniziale_mb:.6f}",
                f"{ram_finale_mb:.6f}",
                f"{delta_ram_mb:.6f}",
                f"{percentuale_ram_sistema:.6f}",
                epoche_locali,
                f"{learning_rate_fit:.8f}",
            ]
        )


# ------------------------------
# METRICA AGGREGATA
# ------------------------------
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}




def get_on_fit_config_fn(base_lr: float, local_epochs: int):
    """Return fit config with explicit round and LR schedule for clients."""

    def fit_config_fn(server_round: int) -> dict[str, Scalar]:
        # Step decay ogni 20 round per migliorare la convergenza
        lr_round = base_lr * (0.5 ** ((server_round - 1) // 20))
        lr_round = max(lr_round, base_lr * 0.05)
        return {
            "current_round": server_round,
            "learning_rate": lr_round,
            "local_epochs": local_epochs,
        }

    return fit_config_fn

def aggregate_fit_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate and print per-client CPU metrics returned by fit()."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    tempo_cpu_pesato = 0.0
    tempo_reale_pesato = 0.0
    core_equivalenti_pesati = 0.0
    percentuale_cpu_pesata = 0.0
    ram_iniziale_mb_pesata = 0.0
    ram_finale_mb_pesata = 0.0
    delta_ram_mb_pesata = 0.0
    percentuale_ram_sistema_pesata = 0.0

    for idx, (num_examples, metric) in enumerate(metrics, start=1):
        tempo_cpu = float(metric.get("tempo_cpu_fit", 0.0))
        tempo_reale = float(metric.get("tempo_reale_fit", 0.0))
        core_equivalenti = float(metric.get("core_equivalenti_fit", 0.0))
        percentuale_cpu = float(metric.get("percentuale_cpu_fit", 0.0))
        ram_iniziale_mb = float(metric.get("ram_iniziale_mb_fit", 0.0))
        ram_finale_mb = float(metric.get("ram_finale_mb_fit", 0.0))
        delta_ram_mb = float(metric.get("delta_ram_mb_fit", 0.0))
        percentuale_ram_sistema = float(metric.get("percentuale_ram_sistema_fit", 0.0))
        fit_round = metric.get("fit_round", "unknown")
        epoche_locali = metric.get("epoche_locali_fit", "n/a")
        learning_rate_fit = float(metric.get("learning_rate_fit", 0.0))

        tempo_cpu_pesato += tempo_cpu * num_examples
        tempo_reale_pesato += tempo_reale * num_examples
        core_equivalenti_pesati += core_equivalenti * num_examples
        percentuale_cpu_pesata += percentuale_cpu * num_examples
        ram_iniziale_mb_pesata += ram_iniziale_mb * num_examples
        ram_finale_mb_pesata += ram_finale_mb * num_examples
        delta_ram_mb_pesata += delta_ram_mb * num_examples
        percentuale_ram_sistema_pesata += percentuale_ram_sistema * num_examples

        log(
            logging.INFO,
            "[Round %s] Client-%s metriche | esempi=%s | tempo_cpu=%.3fs | tempo_reale=%.3fs | core_equivalenti=%.2f | percentuale_cpu=%.2f%% | ram_ini=%.1fMB | ram_fin=%.1fMB | delta_ram=%.1fMB | ram_sistema=%.2f%% | epoche=%s | lr=%.5f",
            fit_round,
            idx,
            num_examples,
            tempo_cpu,
            tempo_reale,
            core_equivalenti,
            percentuale_cpu,
            ram_iniziale_mb,
            ram_finale_mb,
            delta_ram_mb,
            percentuale_ram_sistema,
            epoche_locali,
            learning_rate_fit,
        )

        _append_client_metric_row(
            fit_round,
            idx,
            num_examples,
            tempo_cpu,
            tempo_reale,
            core_equivalenti,
            percentuale_cpu,
            ram_iniziale_mb,
            ram_finale_mb,
            delta_ram_mb,
            percentuale_ram_sistema,
            epoche_locali,
            learning_rate_fit,
        )

    if total_examples == 0:
        return {}

    return {
        "tempo_cpu_fit": tempo_cpu_pesato / total_examples,
        "tempo_reale_fit": tempo_reale_pesato / total_examples,
        "core_equivalenti_fit": core_equivalenti_pesati / total_examples,
        "percentuale_cpu_fit": percentuale_cpu_pesata / total_examples,
        "ram_iniziale_mb_fit": ram_iniziale_mb_pesata / total_examples,
        "ram_finale_mb_fit": ram_finale_mb_pesata / total_examples,
        "delta_ram_mb_fit": delta_ram_mb_pesata / total_examples,
        "percentuale_ram_sistema_fit": percentuale_ram_sistema_pesata / total_examples,
    }


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
        on_fit_config_fn=get_on_fit_config_fn(
            base_lr=float(context.run_config["learning-rate"]),
            local_epochs=int(context.run_config["local-epochs"]),
        ),
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        stop_criteria={"metric_ge": ("accuracy", ACCURACY),
        },
    )
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)
# ------------------------------
# AVVIO SERVER
# ------------------------------
app = ServerApp(server_fn=server_fn)
