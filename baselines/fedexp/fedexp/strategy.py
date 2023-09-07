from typing import List, Tuple, Union, Optional, Dict

import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters

from fedexp.utils import get_parameters


class FedExP(FedAvg):
    def __init__(
            self,
            net_glob: torch.nn.Module,
            epsilon: float,
            decay: float,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.net_glob = net_glob
        self.epsilon = epsilon
        self.decay = decay
        self.w_vec_estimate = np.zeros(parameters_to_vector(self.net_glob.parameters()).numel())
        self.server_steps = [0]

    def __repr__(self) -> str:
        return "FedExP"

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using FedProx."""
        if not self.accept_failures and failures:
            return None, {}
        # Aggregate results
        grad_sum = sum([res.metrics["grad_p"] for _, res in results])
        p_sum = sum([res.metrics["p"] for _, res in results])
        grad_norm_sum = sum([res.metrics["grad_norm"] for _, res in results])
        clients_per_round = len(results) + len(failures)
        with torch.no_grad():
            grad_avg = grad_sum / p_sum
            grad_avg_norm = torch.linalg.norm(grad_avg) ** 2
            grad_norm_avg = grad_norm_sum / p_sum
            eta_g = max(1, (0.5 * grad_norm_avg /
                            (grad_avg_norm + clients_per_round * self.epsilon).cpu()).item())
            self.server_steps.append(eta_g)
            w_vec_prev = self.w_vec_estimate
            self.w_vec_estimate = parameters_to_vector(self.net_glob.parameters()) + eta_g * grad_avg
            w_vec_avg = self.w_vec_estimate if server_round == 0 else (self.w_vec_estimate + w_vec_prev) / 2
        vector_to_parameters(w_vec_avg, self.net_glob.parameters())
        self.epsilon *= self.decay ** 2
        return ndarrays_to_parameters(get_parameters(self.net_glob)), {}
