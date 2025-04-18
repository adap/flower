"""feddefender: A Flower Baseline."""
from collections import Counter

import flwr as fl

from feddefender import utils
from feddefender.differential_testing import differential_testing_fl_clients


class FedAvgWithFedDefender(fl.server.strategy.FedAvg):
    """FedAvg with Differential Testing."""

    def __init__(
        self,
        num_bugs,
        num_inputs,
        input_shape,
        na_t,
        device,
        fast,
        theta,
        callback_create_model_fn,
        callback_fed_defender_evaluate_fn,
        *args,
        **kwargs,
    ):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.input_shape = input_shape
        self.num_bugs = num_bugs
        self.num_inputs = num_inputs
        self.na_t = na_t
        self.device = device
        self.fast = fast
        self.theta = theta
        self.create_model_fn = callback_create_model_fn
        self.callback_fed_defender_evaluate_fn = callback_fed_defender_evaluate_fn

def aggregate_fit(self, server_round, results, failures):
    """Aggregate clients updates using FedDefender defense."""
    # Get weights and number of examples from each client
    client2weights = {
        fit_res.metrics["cid"]: fit_res.parameters
        for _, fit_res in results
    }
    
    client2num_examples = {
        fit_res.metrics["cid"]: fit_res.num_examples
        for _, fit_res in results
    }
    
    min_nk = min(client2num_examples.values())
    
    client2mal_confidence = self._run_differential_testing_helper(results)

    # Calculate Attack Success Rate (ASR)
    detected_malicious = sum(1 for conf in client2mal_confidence.values() if conf > self.theta)
    total_clients = len(client2mal_confidence)
    attack_success_rate = detected_malicious / total_clients if total_clients > 0 else 0.0
    
    for client, confidence in client2mal_confidence.items():
        if confidence > self.theta:
            client2num_examples[client] = 0
        else:
            client2num_examples[client] = int(min_nk * (1 - confidence))
        
        
        for client, confidence in client2mal_confidence.items():
            if confidence > self.theta:
                client2num_examples[client] = 0
            else:
                client2num_examples[client] = int(min_nk * (1 - confidence))
        
        weights_results = [
            (weights, client2num_examples[cid])
            for cid, weights in client2weights.items()
        ]
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, weights_results, failures
        )
        
        aggregated_metrics["malicious_confidence_scores"] = client2mal_confidence
        aggregated_metrics["attack_success_rate"] = attack_success_rate

        self.callback_fed_defender_evaluate_fn(server_round, client2mal_confidence)
        print("Attack Success Rate: ")
        print(attack_success_rate)
        
        return aggregated_parameters, aggregated_metrics

def _run_differential_testing_helper(self, results):
    """Run differential testing and return malicious confidence scores."""
    client2model = {
        fit_res.metrics["cid"]: self._get_model_from_parameters(fit_res.parameters)
        for _, fit_res in results
    }
    
    client2mal_confidence = {cid: 0.0 for cid in client2model.keys()}
    
    for _ in range(self.num_inputs):
        predicted_faulty_clients = differential_testing_fl_clients(
            client2model,
            self.num_bugs,
            1,
            self.input_shape,
            self.na_t,
            self.fast,
            self.device,
        )[0]
        
        for client in predicted_faulty_clients:
            client2mal_confidence[f"{client}"] += 1
    
    for client in client2mal_confidence:
        client2mal_confidence[client] /= self.num_inputs
    
    return client2mal_confidence