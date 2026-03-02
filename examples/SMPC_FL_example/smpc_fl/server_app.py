"""SMPC Federated Learning Server App."""

from typing import List, Tuple, Optional, Dict
import numpy as np
import flwr as fl
from flwr.common import Context, FitRes, EvaluateRes, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy

class SMPCStrategy(FedAvg):
    """SMPC-based Federated Averaging strategy."""

    def __init__(self, fraction_fit: float = 1.0):
        super().__init__(fraction_fit=fraction_fit)
        self.loss_per_round = []
        self.accuracy_per_round = []

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """Initialize with random weights."""
        np.random.seed(42)
        weights1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2 / 784)
        bias1 = np.zeros(128, dtype=np.float32)
        weights2 = np.random.randn(128, 10).astype(np.float32) * np.sqrt(2 / 128)
        bias2 = np.zeros(10, dtype=np.float32)
        return ndarrays_to_parameters([weights1, bias1, weights2, bias2])

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], 
                     failures: List[BaseException]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate using SMPC protocol."""
        if not results:
            return None, {}

        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_samples = [fit_res.num_examples for _, fit_res in results]
        
        total_samples = sum(num_samples)
        aggregated_weights = []

        for layer_weight in zip(*weights_results):
            layer_shape = layer_weight[0].shape
            if layer_weight[0].ndim == 0:
                aggregated_layer = sum(w * n for w, n in zip(layer_weight, num_samples)) / total_samples
                aggregated_weights.append(aggregated_layer)
                continue

            flattened_weights = [np.array(w, dtype=np.float32).flatten() for w in layer_weight]
            aggregated_layers = sum(weight * num_samples[i] for i, weight in enumerate(flattened_weights))
            aggregated_layers /= total_samples
            aggregated_weights.append(aggregated_layers.reshape(layer_shape))

        return ndarrays_to_parameters(aggregated_weights), {}

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, EvaluateRes]], 
                          failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}

        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        losses = [evaluate_res.loss for _, evaluate_res in results]
        num_samples = [evaluate_res.num_examples for _, evaluate_res in results]

        weighted_accuracy = np.average(accuracies, weights=num_samples)
        weighted_loss = np.average(losses, weights=num_samples)

        print(f"Round {server_round}: Accuracy: {weighted_accuracy:.4f}, Loss: {weighted_loss:.4f}")
        self.loss_per_round.append(weighted_loss)
        self.accuracy_per_round.append(weighted_accuracy)

        return weighted_loss, {"accuracy": weighted_accuracy}

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure fit with peer addresses for P2P SMPC."""
        config = {}
        num_clients = client_manager.num_available()
        base_port = 50051
        peer_addresses = [f"localhost:{base_port + i}" for i in range(num_clients)]
        config["peer_addresses"] = ",".join(peer_addresses)
        
        fit_ins = fl.common.FitIns(parameters, config)
        clients = client_manager.sample(num_clients=num_clients, min_num_clients=num_clients)
        return [(client, fit_ins) for client in clients]


def server_fn(context: Context):
    """Construct components for ServerApp."""
    num_rounds = context.run_config.get("num-server-rounds", 10)
    fraction_fit = context.run_config.get("fraction-fit", 1.0)

    strategy = SMPCStrategy(fraction_fit=fraction_fit)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
