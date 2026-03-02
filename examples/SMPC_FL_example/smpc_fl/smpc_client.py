"""SMPC Protocol implementation for secret sharing."""

import numpy as np
from typing import List, Dict


class SMPCProtocol:
    """Handles SMPC secret sharing operations."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients

    def generate_value_secret_share(self, value: float, num_shares: int) -> List[float]:
        """Generate additive secret shares for a value."""
        shares = [np.random.uniform(-1, 1) for _ in range(num_shares - 1)]
        last_share = value - sum(shares)
        shares.append(last_share)
        return shares

    def generate_matrix_secret_shares(self, matrix: np.ndarray, num_shares: int) -> List[np.ndarray]:
        """Generate secret shares for each element in a matrix."""
        shares = [np.zeros_like(matrix) for _ in range(num_shares)]
        for index, value in np.ndenumerate(matrix):
            value_shares = self.generate_value_secret_share(value, num_shares)
            for i in range(num_shares):
                shares[i][index] = value_shares[i]
        return shares

    def split_weights_to_shares(self, weights: List[np.ndarray], num_clients: int) -> Dict[int, List[np.ndarray]]:
        """Split model weights into secret shares for each client."""
        shares_per_weight = [self.generate_matrix_secret_shares(weight, num_clients) for weight in weights]
        shares_grouped_by_client = {i: [shares[i] for shares in shares_per_weight] for i in range(num_clients)}
        return shares_grouped_by_client

    def reconstruct_weights(self, shares_grouped_by_client: List[List[np.ndarray]]) -> List[np.ndarray]:
        """Reconstruct model weights from secret shares."""
        num_weights = len(shares_grouped_by_client[0])
        reconstructed_weights = [
            sum(client_shares[i] for client_shares in shares_grouped_by_client)
            for i in range(num_weights)
        ]
        return reconstructed_weights
