from typing import List, Tuple, Union, Optional, Dict

import torch
from torch.nn.utils import parameters_to_vector
from torch.nn.utils import vector_to_parameters

from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedExP(FedAvg):
    # pass
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        """Aggregate fit results using FedProx."""
        # Aggregate results
        grad_sum = sum([res.metrics["grad_p"] for _, res in results])
        p_sum = sum([res.metrics["p"] for _, res in results])
        grad_norm_sum = sum([res.metrics["grad_norm"] for _, res in results])
        clients_per_round= len(results) +len(failures)

        with torch.no_grad():
            grad_avg = grad_sum/ p_sum
            grad_avg_norm = torch.linalg.norm(grad_avg) ** 2
            grad_norm_avg = grad_norm_sum / p_sum
            
            eta_g = max(1, (0.5 * grad_norm_avg / (grad_avg_norm + clients_per_round * results[1].metrics["epsilon"])).cpu())

            w_vec_prev = w_vec_estimate
            w_vec_estimate = parameters_to_vector(net_glob.parameters()) + eta_g * grad_avg

            w_vec_avg = w_vec_estimate if server_round == 0 else (w_vec_estimate + w_vec_prev) / 2
            vector_to_parameters(w_vec_estimate, net_glob.parameters())

        net_eval = copy.deepcopy(net_glob)
        
        vector_to_parameters(w_vec_avg, net_eval.parameters())

        

        # call the parent `aggregate_fit()` (i.e. that in standard FedAvg)
        return super().aggregate_fit(server_round, results, failures)




        # parameters, num_examples = super().aggregate_fit(server_round,
        #                                                  results,
        #                                                  failures)
    
        # # Return updated parameters
        # return parameters, {"num_examples": num_examples}

