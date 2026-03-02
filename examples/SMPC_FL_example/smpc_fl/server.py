import argparse
import threading
from typing import List, Tuple, Optional, Dict
import time
import flwr as fl
import numpy as np
from flwr.common import FitRes, EvaluateRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from utils import plot_metrics

class SMPCServer(fl.server.strategy.FedAvg):
    """
    A federated learning server that uses Secure Multi-Party Computation (SMPC) to aggregate weights.

    Attributes:
        num_clients (int): Number of clients expected to connect.
        clients (list): List of connected clients.
        wait_timeout (int): Timeout in seconds to wait for clients to connect.
        clients_ready_event (threading.Event): Event to signal when all clients are connected.
        model_structure (None): Placeholder for model structure.
        loss_per_round (list): List to store loss per round.
        accuracy_per_round (list): List to store accuracy per round.
        Initialize the SMPCServer with the number of clients.
    """
    def __init__(self, num_clients: int):
        """
        Initialize the SMPCServer with the specified number of clients.

        Args:
            num_clients (int): The number of clients expected to participate in training.
        """
        super().__init__()
        self.num_clients = num_clients
        self.clients = []
        self.wait_timeout = 60 # seconds
        self.clients_ready_event = threading.Event()
        self.model_structure = None
        self.loss_per_round = []
        self.accuracy_per_round = []

    def initialize_parameters(self, client_manager: fl.server.client_manager.ClientManager) -> Optional[Parameters]:
        """Ensure the server waits for all clients to connect before initializing parameters"""
        print(f"Waiting for {self.num_clients} clients to connect...")
        net_start = time.time()
        net_time = 0
        client_manager.wait_for(num_clients=self.num_clients, timeout=self.wait_timeout)
        net_end = time.time()
        net_time = net_end - net_start
        print(f"-- Time taken to connect all clients: {net_time} seconds --")
        self.clients_ready_event.set()
        print("All clients connected. Initializing parameters...")
        return fl.common.ndarrays_to_parameters(self.generate_initial_weights())

    def generate_initial_weights(self) -> List[np.ndarray]:
        """
        Generate initial random weights for the model.

        Returns:
            List[np.ndarray]: A list of numpy arrays representing the model's initial weights.
        """
        np.random.seed(42) # for reprodcibility

        # Layer 1: Dense 128
        weights1 = np.random.randn(784, 128).astype(np.float32) * np.sqrt(2 / 784)
        bias1 = np.zeros(128, dtype=np.float32)

        # Layer 2: Dense 10
        weights2 = np.random.randn(128, 10).astype(np.float32) * np.sqrt(2 / 128)
        bias2 = np.zeros(10, dtype=np.float32)

        return [weights1, bias1, weights2, bias2]

    def aggregate_smpc_weights(self, weights_results: List[List[np.ndarray]],
                               num_samples: List[int]) -> List[np.ndarray]:
        """
        Aggregate weights using Secure Multi-Party Computation (SMPC).

        Args:
            weights_results (List[List[np.ndarray]]): The list of weight updates from clients.
            num_samples (List[int]): The number of samples used by each client during training.

        Returns:
            List[np.ndarray]: The aggregated weights.
        """
        total_samples = sum(num_samples)
        aggregated_weights = []

        for layer_weight in zip(*weights_results):
            layer_shape = layer_weight[0].shape

            if layer_weight[0].ndim == 0:
                # Bias
                aggregated_layer = sum(w * n for w, n in zip(layer_weight, num_samples)) / total_samples
                aggregated_weights.append(aggregated_layer)
                continue

            flattened_weights = [np.array(w, dtype=np.float32).flatten() for w in layer_weight]

            # Sum up the shares
            aggregated_layers = sum(weight * num_samples[i] for i, weight in enumerate(flattened_weights))
            aggregated_layers /= total_samples  # Normalize by total samples

            aggregated_layer = aggregated_layers.reshape(layer_shape)
            aggregated_weights.append(aggregated_layer)

        return aggregated_weights

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[BaseException]) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
        """
        Aggregate the results of the clients' training using the SMPC protocol.

        Args:
            server_round (int): The current round of training.
            results (List[Tuple[ClientProxy, FitRes]]): A list of tuples containing the client proxy and fit results.
            failures (List[BaseException]): A list of exceptions raised during the training process.

        Returns:
            Tuple[Optional[Parameters], Dict[str, Scalar]]: The aggregated model parameters and additional metrics.
        """
        if not results:
            return None, {}

        weights_results = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        num_samples = [fit_res.num_examples for _, fit_res in results]

        # Aggregate weights using SMPC
        aggregated_weights = self.aggregate_smpc_weights(weights_results, num_samples)
        assert len(aggregated_weights) == len(weights_results[0]), "Mismatch in number of layers"

        # Aggregate metrics (if any)
        parameters = fl.common.ndarrays_to_parameters(aggregated_weights)
        metrics_aggregated = {}
        return parameters, metrics_aggregated

    def aggregate_evaluate(self, server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[BaseException]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Aggregate the results of the clients' evaluation using the SMPC protocol.

        Args:
            server_round (int): The current round of evaluation.
            results (List[Tuple[ClientProxy, EvaluateRes]]): A list of tuples containing the client proxy and evaluation results.
            failures (List[BaseException]): A list of exceptions raised during the evaluation process.

        Returns:
            Tuple[Optional[float], Dict[str, Scalar]]: The aggregated loss and additional metrics.
        """
        if not results:
            print(f"Round {server_round}: No successfull evaluations. Failures: {failures}")
            return None, {}

        accuracies = [evaluate_res.metrics["accuracy"] for _, evaluate_res in results]
        losses = [evaluate_res.loss for _, evaluate_res in results]
        num_samples = [evaluate_res.num_examples for _, evaluate_res in results]

        # Calculate weighted average of accuracies and losses
        weighted_accuracy = np.average(accuracies, weights=num_samples)
        weighted_loss = np.average(losses, weights=num_samples)

        print(f"Round {server_round}: Weighted Accuracy: {weighted_accuracy}, Weighted Loss: {weighted_loss}")
        self.loss_per_round.append(weighted_loss)
        self.accuracy_per_round.append(weighted_accuracy)
        # plot_metrics(self.loss_per_round, self.accuracy_per_round)
        if server_round == 15:
            plot_metrics(self.loss_per_round, self.accuracy_per_round)

        return weighted_loss, {"accuracy": weighted_accuracy}

    def configure_fit(self, server_round: int,
                      parameters: Parameters,
                      client_manager: fl.server.client_manager.ClientManager) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """
        Wait for all clients to connect before configuring the training instructions.

        Args:
            server_round (int): The current round of training.
            parameters (Parameters): The model parameters to be sent to the clients.
            client_manager (fl.server.client_manager.ClientManager): The client manager handling connected clients.

        Returns:
            List[Tuple[ClientProxy, fl.common.FitIns]]: A list of tuples containing the client proxy and fit instructions.
        """
        print(f"Round {server_round}: Waiting for all clients to connect...")
        if server_round == 1:
            # in the fits round, we wait for all clients to connect
            if not self.clients_ready_event.wait(self.wait_timeout):
                print(f"Warning: Not all clients connected after {self.wait_timeout} seconds")

        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        return fit_ins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Server")
    parser.add_argument("--num_clients", type=int, required=True, help="Number of clients")
    parser.add_argument("--num_rounds", type=int, required=False, help="Number of rounds", default=10)
    args = parser.parse_args()

    strategy = SMPCServer(num_clients=args.num_clients)
    start_time = time.time()
    fl.server.start_server(server_address="localhost:8080",
                           strategy=strategy,
                           config=fl.server.ServerConfig(num_rounds=args.num_rounds))
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
